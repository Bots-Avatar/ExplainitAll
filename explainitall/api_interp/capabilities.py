from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional

from .openrouter_client import OpenRouterClient, ProviderSettings

# Чем меньше ранг, тем ближе веса провайдера к исходной модели
QUANTIZATION_RANK = {'fp32': 0, 'bf16': 1, 'fp16': 1, 'fp8': 2, 'int8': 2, 'fp6': 3, 'fp4': 4, 'int4': 4}
UNKNOWN_QUANTIZATION_RANK = 5

LOGPROB_PARAMETERS = ('logprobs', 'top_logprobs')


def _price(pricing: Optional[dict], key: str) -> float:
    try:
        return float((pricing or {}).get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    hugging_face_id: Optional[str]
    supported_parameters: FrozenSet[str]
    context_length: Optional[int]
    prompt_price: float
    completion_price: float

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelInfo':
        return cls(id=d['id'],
                   name=d.get('name') or d['id'],
                   hugging_face_id=d.get('hugging_face_id') or None,
                   supported_parameters=frozenset(d.get('supported_parameters') or ()),
                   context_length=d.get('context_length'),
                   prompt_price=_price(d.get('pricing'), 'prompt'),
                   completion_price=_price(d.get('pricing'), 'completion'))

    def supports(self, parameters: Iterable[str]) -> bool:
        """Поддержка у модели — объединение по всем провайдерам; конкретного провайдера проверяйте по эндпоинтам"""
        return set(parameters) <= self.supported_parameters


@dataclass(frozen=True)
class EndpointInfo:
    provider_name: str
    tag: str
    quantization: Optional[str]
    supported_parameters: FrozenSet[str]
    status: int
    uptime_last_30m: Optional[float]
    context_length: Optional[int]
    max_completion_tokens: Optional[int]
    prompt_price: float
    completion_price: float

    @classmethod
    def from_dict(cls, d: dict) -> 'EndpointInfo':
        return cls(provider_name=d.get('provider_name') or d.get('tag') or '',
                   tag=d.get('tag') or d.get('provider_name') or '',
                   quantization=d.get('quantization'),
                   supported_parameters=frozenset(d.get('supported_parameters') or ()),
                   status=int(d.get('status') or 0),
                   uptime_last_30m=d.get('uptime_last_30m'),
                   context_length=d.get('context_length'),
                   max_completion_tokens=d.get('max_completion_tokens'),
                   prompt_price=_price(d.get('pricing'), 'prompt'),
                   completion_price=_price(d.get('pricing'), 'completion'))

    @property
    def available(self) -> bool:
        """Отрицательный статус означает, что эндпоинт сейчас недоступен или работает со сбоями"""
        return self.status >= 0

    @property
    def quantization_rank(self) -> int:
        return QUANTIZATION_RANK.get((self.quantization or '').lower(), UNKNOWN_QUANTIZATION_RANK)

    def supports(self, parameters: Iterable[str]) -> bool:
        return set(parameters) <= self.supported_parameters

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, n_requests: int = 1) -> float:
        """Оценка стоимости серии одинаковых по размеру запросов в долларах"""
        return n_requests * (prompt_tokens * self.prompt_price + completion_tokens * self.completion_price)


def get_models(client: OpenRouterClient, required_parameters: Iterable[str] = (),
               output_modalities: Optional[Iterable[str]] = None) -> List[ModelInfo]:
    """
    Модели, у которых хотя бы один провайдер поддерживает все required_parameters.
    output_modalities: тип моделей, например ['embeddings'] или ['rerank']; по умолчанию — текстовые
    """
    required = list(required_parameters)
    if output_modalities:
        raw = client.list_models(required or None, output_modalities=list(output_modalities))
    else:
        raw = client.list_models(required or None)
    return [m for m in (ModelInfo.from_dict(d) for d in raw) if m.supports(required)]


def get_endpoints(client: OpenRouterClient, model: str) -> List[EndpointInfo]:
    return [EndpointInfo.from_dict(d) for d in client.list_endpoints(model).get('endpoints') or []]


def select_endpoint(endpoints: Iterable[EndpointInfo],
                    required: Iterable[str] = LOGPROB_PARAMETERS,
                    preferred: Iterable[str] = ('seed',),
                    allowed_quantizations: Optional[Iterable[str]] = None,
                    include_unavailable: bool = False) -> Optional[EndpointInfo]:
    """
    Выбор провайдера для эксперимента.
    Обязательны все required; среди подходящих выигрывает тот, кто поддерживает больше preferred,
    затем более точные веса, затем более высокая доступность, затем более низкая цена.
    """
    required = list(required)
    preferred = list(preferred)
    allowed = {q.lower() for q in allowed_quantizations} if allowed_quantizations is not None else None

    candidates = [e for e in endpoints
                  if (include_unavailable or e.available)
                  and e.supports(required)
                  and (allowed is None or (e.quantization or '').lower() in allowed)]
    if not candidates:
        return None

    def rank(e: EndpointInfo):
        preferred_supported = sum(p in e.supported_parameters for p in preferred)
        uptime = e.uptime_last_30m if e.uptime_last_30m is not None else 0.0
        return -preferred_supported, e.quantization_rank, -uptime, e.prompt_price + e.completion_price

    return min(candidates, key=rank)


def pinned_provider(endpoint: EndpointInfo,
                    data_collection: Optional[str] = None,
                    zdr: Optional[bool] = None) -> ProviderSettings:
    """
    Маршрутизация строго на выбранный эндпоинт: без подмены провайдера
    и без молчаливого отбрасывания неподдерживаемых параметров.
    """
    return ProviderSettings(order=[endpoint.tag], allow_fallbacks=False, require_parameters=True,
                            data_collection=data_collection, zdr=zdr)
