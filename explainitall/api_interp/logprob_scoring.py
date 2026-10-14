from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .openrouter_client import ChatResult, OpenRouterClient, OpenRouterError, ProviderSettings, TokenLogprob


@dataclass
class PrefillCheck:
    supported: Optional[bool]  # None — эталон слишком короткий для проверки
    n_calls: int = 0
    n_cached: int = 0
    cost: float = 0.0


@dataclass
class ForcedScore:
    """
    Логарифм вероятности эталонного ответа при заданном промпте.
    Если эталонного токена не было среди top_logprobs, вместо его logprob берётся наименьший
    из полученных — это верхняя оценка, и такие токены считаются в n_truncated.
    """
    token_logprobs: List[float] = field(default_factory=list)
    n_reference_tokens: int = 0
    n_truncated: int = 0
    n_calls: int = 0
    n_cached: int = 0
    cost: float = 0.0

    @property
    def n_tokens(self) -> int:
        return len(self.token_logprobs)

    @property
    def logprob(self) -> float:
        return float(sum(self.token_logprobs))

    @property
    def complete(self) -> bool:
        return self.n_tokens >= self.n_reference_tokens


def _lookup(step: TokenLogprob, target: str):
    """logprob токена target на шаге step; (значение, усечено ли)"""
    for token, logprob in step.top_logprobs:
        if token == target:
            return logprob, False
    if step.top_logprobs:
        return min(lp for _, lp in step.top_logprobs), True
    return step.logprob, True


def consume_generation(steps: Sequence[TokenLogprob], reference: Sequence[str], position: int, score: ForcedScore) -> int:
    """
    Идёт по сгенерированным токенам, пока они совпадают с эталоном, записывая их logprob.
    На первом расхождении записывает logprob эталонного токена и останавливается.
    Возвращает новую позицию в эталоне.
    """
    for step in steps:
        if position >= len(reference):
            break
        target = reference[position]
        if step.token == target:
            score.token_logprobs.append(step.logprob)
            position += 1
            continue
        logprob, truncated = _lookup(step, target)
        score.token_logprobs.append(logprob)
        score.n_truncated += int(truncated)
        position += 1
        break
    return position


class ForcedLogprobScorer:
    """
    Оценка log p(эталонный ответ | промпт) для модели за API без teacher forcing.

    Генерация идёт с top_logprobs; пока модель повторяет эталон, logprob точен.
    На расхождении берётся logprob эталонного токена из top_logprobs и генерация
    перезапускается с prefill — эталонным началом ответа в сообщении ассистента.
    Запросов: 1 + число расхождений (не больше max_calls).
    Без поддержки prefill (use_prefill=False) оценивается только начало ответа до первого расхождения.
    """

    def __init__(self,
                 client: OpenRouterClient,
                 model: str,
                 provider: Optional[ProviderSettings] = None,
                 top_logprobs: int = 20,
                 max_answer_tokens: int = 32,
                 max_calls: int = 8,
                 temperature: Optional[float] = 0.0,
                 seed: Optional[int] = None,
                 use_prefill: bool = True):
        self.client = client
        self.model = model
        self.provider = provider
        self.top_logprobs = top_logprobs
        self.max_answer_tokens = max_answer_tokens
        self.max_calls = max_calls
        self.temperature = temperature
        self.seed = seed
        self.use_prefill = use_prefill

    def reference_tokens(self, result: ChatResult) -> List[str]:
        """Эталонные токены из ответа, полученного с logprobs"""
        if result.logprobs is None:
            raise OpenRouterError('Провайдер не вернул logprobs: закрепите провайдера с поддержкой logprobs '
                                  '(capabilities.select_endpoint + pinned_provider)')
        return [t.token for t in result.logprobs[:self.max_answer_tokens]]

    def _call(self, messages: List[dict], prefix: Sequence[str], max_tokens: int) -> ChatResult:
        request = list(messages)
        if prefix:
            request = request + [{'role': 'assistant', 'content': ''.join(prefix)}]
        return self.client.chat(request, self.model, temperature=self.temperature, max_tokens=max_tokens,
                                seed=self.seed, top_logprobs=self.top_logprobs, provider=self.provider)

    def score(self, messages: List[dict], reference: Sequence[str],
              first_result: Optional[ChatResult] = None) -> ForcedScore:
        """
        messages: промпт без ответа
        reference: эталонные токены ответа
        first_result: уже полученная генерация по этому промпту с теми же параметрами (экономит запрос)
        """
        reference = list(reference)[:self.max_answer_tokens]
        score = ForcedScore(n_reference_tokens=len(reference))
        position = 0

        if first_result is not None:
            if first_result.logprobs is None:
                raise OpenRouterError('В переданной генерации нет logprobs')
            position = consume_generation(first_result.logprobs, reference, position, score)
            if not self.use_prefill:
                return score

        while position < len(reference) and score.n_calls < self.max_calls:
            if position > 0 and not self.use_prefill:
                break
            result = self._call(messages, reference[:position], len(reference) - position)
            score.n_calls += 1
            if result.cached:
                score.n_cached += 1
            else:
                score.cost += result.cost
            if result.logprobs is None:
                raise OpenRouterError('Провайдер не вернул logprobs')
            if not result.logprobs:
                break  # модель закончила ответ там, где эталон продолжается: распределение недоступно
            position = consume_generation(result.logprobs, reference, position, score)
            if not self.use_prefill:
                break
        return score

    def verify_prefill(self, messages: List[dict], reference: Sequence[str]) -> PrefillCheck:
        """
        Проверка, что провайдер продолжает ответ с prefill, а не начинает заново.
        По исходному промпту при temperature=0 модель после prefill эталонного начала
        должна выдать следующий эталонный токен.
        """
        reference = list(reference)[:self.max_answer_tokens]
        positions = sorted({p for p in (len(reference) // 3, 2 * len(reference) // 3) if 1 <= p < len(reference)})
        check = PrefillCheck(supported=None if not positions else True)
        for position in positions:
            result = self._call(messages, reference[:position], max_tokens=1)
            check.n_calls += 1
            if result.cached:
                check.n_cached += 1
            else:
                check.cost += result.cost
            if not result.logprobs or result.logprobs[0].token != reference[position]:
                check.supported = False
                break
        return check
