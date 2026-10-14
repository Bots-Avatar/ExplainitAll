import re
import threading

import numpy as np
import pytest

from explainitall.api_interp.logprob_scoring import ForcedLogprobScorer, consume_generation, ForcedScore
from explainitall.api_interp.openrouter_client import ChatResult, OpenRouterError, TokenLogprob

PROMPT = 'У моей кошки грипп. Чем лечить кошку?'
NO_DISEASE = 'У моей кошки [...]. Чем лечить кошку?'
NO_ANIMAL = 'У моей [...] грипп. Чем лечить [...]?'

PATH_FULL = [' Кошке', ' при', ' гриппе', ' помогут', ' уколы']
PATH_NO_DISEASE = [' Кошке', ' нужен', ' осмотр', ' ветеринара']
PATH_NO_ANIMAL = [' При', ' гриппе', ' помогут', ' уколы']
ALL_PATHS = [PATH_FULL, PATH_NO_DISEASE, PATH_NO_ANIMAL]

ON_PATH, OFF_PATH, ALTERNATIVE = -0.05, -0.5, -3.0


def path_for(prompt):
    has_animal, has_disease = 'кошк' in prompt, 'грипп' in prompt
    if has_animal and has_disease:
        return PATH_FULL
    return PATH_NO_DISEASE if has_animal else PATH_NO_ANIMAL


class FakeTokenLM:
    """
    Модель с жадной генерацией и logprobs.
    Пока префикс ответа совпадает с её собственным путём, она уверенно продолжает его (ON_PATH);
    после навязанного чужого префикса продолжает исходный ответ PATH_FULL менее уверенно (OFF_PATH).
    Токены других путей на той же позиции получают ALTERNATIVE и ниже.
    """

    def __init__(self, prefill=True, logprobs=True, jitter=None):
        self.prefill = prefill
        self.logprobs = logprobs
        self.jitter = jitter or {}
        self.calls = []
        self._lock = threading.Lock()

    def chat(self, messages, model, temperature=None, max_tokens=None, seed=None, top_logprobs=None,
             provider=None, sample_id=0, **kwargs):
        prompt = ' '.join(m['content'] for m in messages if m['role'] != 'assistant')
        prefix_text = messages[-1]['content'] if messages[-1]['role'] == 'assistant' else ''
        with self._lock:
            self.calls.append({'prompt': prompt, 'prefix': prefix_text, 'max_tokens': max_tokens,
                               'top_logprobs': top_logprobs, 'sample_id': sample_id})

        prefix = re.findall(r' [^ ]+', prefix_text) if self.prefill else []
        own_path = path_for(prompt)
        delta = self.jitter.get(sample_id, 0.0) if not prefix_text else 0.0
        steps = []
        for _ in range(max_tokens or 256):
            position = len(prefix)
            if prefix == own_path[:position]:
                path, confidence = own_path, ON_PATH
            else:
                path, confidence = PATH_FULL, OFF_PATH
            if position >= len(path):
                break
            chosen = path[position]
            alternatives = [p[position] for p in ALL_PATHS if position < len(p) and p[position] != chosen]
            top = [(chosen, confidence + delta)] + [(t, ALTERNATIVE - i) for i, t in
                                                    enumerate(dict.fromkeys(alternatives))]
            steps.append(TokenLogprob(chosen, confidence + delta, top[:top_logprobs or 0]))
            prefix = prefix + [chosen]

        generated = steps
        return ChatResult(text=''.join(s.token for s in generated), finish_reason='stop', model=model,
                          provider='Novita', usage={'cost': 0.001}, cost=0.001,
                          logprobs=generated if self.logprobs and top_logprobs is not None else None,
                          reasoning=None, cached=False, raw={})


def messages(prompt):
    return [{'role': 'user', 'content': prompt}]


def make_scorer(lm=None, **kwargs):
    return ForcedLogprobScorer(lm or FakeTokenLM(), 'meta-llama/llama-3.1-8b-instruct', **kwargs)


def test_reference_scored_exactly_in_one_call():
    lm = FakeTokenLM()
    score = make_scorer(lm).score(messages(PROMPT), PATH_FULL)

    assert score.token_logprobs == pytest.approx([ON_PATH] * 5)
    assert score.complete and score.n_calls == 1 and score.n_truncated == 0
    assert lm.calls[0]['top_logprobs'] == 20 and lm.calls[0]['max_tokens'] == 5


def test_divergence_restarts_with_prefill():
    lm = FakeTokenLM()
    score = make_scorer(lm).score(messages(NO_DISEASE), PATH_FULL)

    assert score.token_logprobs == pytest.approx([ON_PATH, ALTERNATIVE, OFF_PATH, OFF_PATH, OFF_PATH])
    assert score.logprob == pytest.approx(-4.55)
    assert score.n_calls == 2 and score.complete
    assert lm.calls[1]['prefix'] == ' Кошке при'
    assert lm.calls[1]['max_tokens'] == 3


def test_divergence_at_first_token():
    score = make_scorer().score(messages(NO_ANIMAL), PATH_FULL)

    assert score.token_logprobs == pytest.approx([ALTERNATIVE] + [OFF_PATH] * 4)
    assert score.n_calls == 2


def test_first_result_saves_a_request():
    lm = FakeTokenLM()
    scorer = make_scorer(lm)
    generation = lm.chat(messages(NO_DISEASE), 'm', max_tokens=256, top_logprobs=20)
    lm.calls.clear()

    score = scorer.score(messages(NO_DISEASE), PATH_FULL, first_result=generation)

    assert score.logprob == pytest.approx(-4.55)
    assert score.n_calls == 1
    assert lm.calls[0]['prefix'] == ' Кошке при'


def test_without_prefill_only_prefix_until_divergence_is_scored():
    lm = FakeTokenLM()
    score = make_scorer(lm, use_prefill=False).score(messages(NO_DISEASE), PATH_FULL)

    assert score.token_logprobs == pytest.approx([ON_PATH, ALTERNATIVE])
    assert not score.complete
    assert len(lm.calls) == 1


def test_max_calls_limits_restarts():
    score = make_scorer(max_calls=1).score(messages(NO_DISEASE), PATH_FULL)

    assert score.n_tokens == 2 and not score.complete


def test_reference_token_outside_top_logprobs_is_truncated():
    score = make_scorer(top_logprobs=1).score(messages(NO_DISEASE), PATH_FULL)

    assert score.n_truncated == 1
    assert score.token_logprobs[1] == pytest.approx(ON_PATH)  # верхняя оценка: наименьший из полученных


def test_max_answer_tokens_limits_reference():
    score = make_scorer(max_answer_tokens=2).score(messages(PROMPT), PATH_FULL)

    assert score.n_reference_tokens == 2 and score.n_tokens == 2


def test_model_stopping_early_leaves_score_incomplete():
    score = make_scorer().score(messages(PROMPT), PATH_FULL + [' быстро'])

    assert score.n_tokens == 5 and not score.complete
    assert score.n_calls == 2


def test_reference_tokens_require_logprobs():
    result = FakeTokenLM(logprobs=False).chat(messages(PROMPT), 'm', top_logprobs=20)

    with pytest.raises(OpenRouterError):
        make_scorer().reference_tokens(result)


def test_verify_prefill():
    supported = make_scorer(FakeTokenLM(prefill=True)).verify_prefill(messages(PROMPT), PATH_FULL)
    ignored = make_scorer(FakeTokenLM(prefill=False)).verify_prefill(messages(PROMPT), PATH_FULL)
    short = make_scorer().verify_prefill(messages(PROMPT), PATH_FULL[:1])

    assert supported.supported is True and supported.n_calls == 2
    assert supported.cost == pytest.approx(0.002)
    assert ignored.supported is False and ignored.n_calls == 1
    assert short.supported is None and short.n_calls == 0


def test_consume_generation_stops_at_divergence():
    score = ForcedScore(n_reference_tokens=3)
    steps = [TokenLogprob('a', -0.1, [('a', -0.1)]), TokenLogprob('x', -0.2, [('x', -0.2), ('b', -1.5)]),
             TokenLogprob('c', -0.3, [])]

    assert consume_generation(steps, ['a', 'b', 'c'], 0, score) == 2
    assert score.token_logprobs == pytest.approx([-0.1, -1.5])
