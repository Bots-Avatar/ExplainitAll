import re
import threading

import numpy as np
import pytest

gensim = pytest.importorskip("gensim")
pytest.importorskip("pymorphy3")

from explainitall.api_interp.cache import ResponseCache
from explainitall.api_interp.explainer import ExplainerAPI
from explainitall.api_interp.openrouter_client import BudgetExceededError, ChatResult, OpenRouterClient
from explainitall.api_interp.perturbations import find_words, replace_spans

PROMPT = 'У моей кошки грипп. Чем лечить кошку?'
CLUSTERS = [{'name': 'Животные', 'centroid': ['кошка'], 'top_k': 1},
            {'name': 'Болезни', 'centroid': ['грипп'], 'top_k': 1},
            {'name': 'Лекарства', 'centroid': ['укол'], 'top_k': 1}]

FULL_ANSWER = 'Кошке при гриппе помогут уколы'
NO_DISEASE_ANSWER = 'Кошке нужен осмотр ветеринара'
NO_ANIMAL_ANSWER = 'При гриппе помогут уколы'


def answer_for(prompt):
    has_animal, has_disease = 'кошк' in prompt, 'грипп' in prompt
    if has_animal and has_disease:
        return FULL_ANSWER
    if has_animal:
        return NO_DISEASE_ANSWER
    return NO_ANIMAL_ANSWER


class FakeChatClient:
    """Детерминированная «модель»: ответ зависит только от того, какие слова остались в промпте"""

    def __init__(self, answer=answer_for, fail_on=None):
        self.answer = answer
        self.fail_on = fail_on
        self.calls = []
        self._lock = threading.Lock()

    def chat(self, messages, model, temperature=None, max_tokens=None, seed=None, provider=None, sample_id=0,
             **kwargs):
        prompt = ' '.join(m['content'] for m in messages)
        with self._lock:
            self.calls.append({'prompt': prompt, 'sample_id': sample_id, 'model': model, 'seed': seed})
        if self.fail_on and self.fail_on in prompt:
            raise BudgetExceededError('Исчерпан бюджет')
        text = self.answer(prompt) if self.answer.__code__.co_argcount == 1 else self.answer(prompt, sample_id)
        return ChatResult(text=text, finish_reason='stop', model=model, provider='Novita', usage={}, cost=0.001,
                          logprobs=None, reasoning=None, cached=False, raw={})


class BagOfWordsEmbedder:
    def __init__(self, dim=128):
        self.dim = dim
        self.vocab = {}
        self._lock = threading.Lock()

    def encode(self, sentences):
        vectors = np.zeros((len(sentences), self.dim))
        with self._lock:
            for i, sentence in enumerate(sentences):
                for word in re.findall(r'\w+', sentence.lower()):
                    vectors[i, self.vocab.setdefault(word, len(self.vocab))] += 1
        return vectors


@pytest.fixture(scope="module")
def nlp_model():
    words = {'кошка_NOUN': [1, 0, 0, 0], 'собака_NOUN': [0.9, 0.1, 0, 0],
             'грипп_NOUN': [0, 1, 0, 0], 'простуда_NOUN': [0, 0.9, 0.1, 0],
             'укол_NOUN': [0, 0, 1, 0], 'таблетка_NOUN': [0, 0, 0.9, 0.1],
             'лечить_VERB': [0, 0, 0, 1]}
    kv = gensim.models.KeyedVectors(vector_size=4)
    kv.add_vectors(list(words), np.array(list(words.values()), dtype=np.float32))
    return kv


def make_explainer(nlp_model, client=None, **kwargs):
    return ExplainerAPI(client=client or FakeChatClient(), model='meta-llama/llama-3.1-8b-instruct',
                        nlp_model=nlp_model, embedder=BagOfWordsEmbedder(), **kwargs)


def row(df, cluster):
    return df[df['cluster'] == cluster].iloc[0]


def test_find_words_keeps_hyphenated_words():
    words = [s.word for s in find_words('Он чуть-чуть болен, кто-то сказал.')]

    assert words == ['Он', 'чуть-чуть', 'болен', 'кто-то', 'сказал']


@pytest.mark.parametrize('replacement,expected', [
    ('[...]', 'У кошки [...], что делать?'),
    ('', 'У кошки, что делать?'),
])
def test_replace_spans(replacement, expected):
    text = 'У кошки грипп, что делать?'
    spans = [s for s in find_words(text) if s.word == 'грипп']

    assert replace_spans(text, spans, replacement) == expected


def test_plan_finds_cluster_words_without_api_calls(nlp_model):
    client = FakeChatClient()
    plan = make_explainer(nlp_model, client).plan(PROMPT, CLUSTERS)

    assert plan.cluster_words == {'Животные': ['кошки', 'кошку'], 'Болезни': ['грипп'], 'Лекарства': []}
    assert list(plan.perturbed) == ['Животные', 'Болезни']
    assert plan.perturbed['Животные'][0]['content'] == 'У моей [...] грипп. Чем лечить [...]?'
    assert plan.n_requests == 3 + 2
    assert client.calls == []


def test_interpret_importance_and_influence(nlp_model):
    client = FakeChatClient()
    out = make_explainer(nlp_model, client).interpret(PROMPT, CLUSTERS)

    assert len(client.calls) == 5
    assert sorted(c['sample_id'] for c in client.calls if 'кошк' in c['prompt'] and 'грипп' in c['prompt']) == [0, 1, 2]
    assert out.reference_answer == FULL_ANSWER
    assert out.noise_mean == out.noise_max == 0.0

    disease, animals, drugs = row(out.importance_df, 'Болезни'), row(out.importance_df, 'Животные'), row(
        out.importance_df, 'Лекарства')
    assert disease['importance'] > animals['importance'] > 0
    assert disease['significant'] and animals['significant']
    assert disease['words'] == ['грипп']
    assert np.isnan(drugs['importance']) and drugs['n_samples'] == 0 and not drugs['significant']
    assert list(out.importance_df['cluster']) == ['Болезни', 'Животные', 'Лекарства']

    influence = out.cluster_imp_aggr_df.set_index('Tokens')
    assert list(influence.columns) == ['Животные', 'Болезни', 'Лекарства']
    assert influence.loc['Болезни'].to_dict() == {'Животные': 0.0, 'Болезни': 1.0, 'Лекарства': 1.0}
    assert influence.loc['Животные'].to_dict() == {'Животные': 1.0, 'Болезни': 0.0, 'Лекарства': 0.0}

    assert out.answers_without('Болезни') == [NO_DISEASE_ANSWER]
    assert out.total_cost == pytest.approx(0.005)
    assert out.n_api_requests == 5


def test_noise_reduces_importance_and_significance(nlp_model):
    def noisy(prompt, sample_id):
        text = answer_for(prompt)
        return text.replace('уколы', 'таблетки') if sample_id == 1 and text == FULL_ANSWER else text

    out = make_explainer(nlp_model, FakeChatClient(noisy), n_baseline=3).interpret(PROMPT, CLUSTERS)

    assert out.noise_max > out.noise_mean > 0
    animals = row(out.importance_df, 'Животные')
    assert animals['importance'] == pytest.approx(animals['distance'] - out.noise_mean)
    assert row(out.importance_df, 'Болезни')['significant']


def test_effect_below_noise_is_not_significant(nlp_model):
    def noisy(prompt, sample_id):
        if 'грипп' in prompt and 'кошк' in prompt:
            return FULL_ANSWER if sample_id == 0 else 'Совсем другой ответ без пересечений'
        return answer_for(prompt)

    out = make_explainer(nlp_model, FakeChatClient(noisy), n_baseline=2).interpret(PROMPT, CLUSTERS)

    assert out.noise_max == pytest.approx(1.0)
    assert not row(out.importance_df, 'Животные')['significant']
    assert row(out.importance_df, 'Животные')['importance'] == 0.0


def test_repeats_and_records_order_do_not_depend_on_workers(nlp_model):
    sequential = make_explainer(nlp_model, max_workers=1, n_repeats=2).interpret(PROMPT, CLUSTERS)
    parallel = make_explainer(nlp_model, max_workers=8, n_repeats=2).interpret(PROMPT, CLUSTERS)

    assert [(r.kind, r.sample_id) for r in sequential.records] == [(r.kind, r.sample_id) for r in parallel.records]
    assert [r.kind for r in sequential.records] == ['baseline'] * 3 + ['Животные'] * 2 + ['Болезни'] * 2
    assert row(sequential.importance_df, 'Болезни')['n_samples'] == 2


def test_system_message_is_not_masked_by_default(nlp_model):
    messages = [{'role': 'system', 'content': 'Ты ветеринар, лечишь кошку.'}, {'role': 'user', 'content': PROMPT}]
    plan = make_explainer(nlp_model).plan(messages, CLUSTERS)

    assert plan.perturbed['Животные'][0]['content'] == messages[0]['content']
    assert 'кошк' not in plan.perturbed['Животные'][1]['content']


def test_nothing_to_mask_raises_without_requests(nlp_model):
    client = FakeChatClient()

    with pytest.raises(ValueError):
        make_explainer(nlp_model, client).interpret('Какая сегодня погода?', CLUSTERS)
    assert client.calls == []


def test_errors_are_propagated(nlp_model):
    with pytest.raises(BudgetExceededError):
        make_explainer(nlp_model, FakeChatClient(fail_on='[...] грипп')).interpret(PROMPT, CLUSTERS)


def test_invalid_sample_counts(nlp_model):
    with pytest.raises(ValueError):
        make_explainer(nlp_model, n_baseline=0)


class PromptAwareSession:
    """HTTP-сессия OpenRouter, отвечающая по содержимому промпта"""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def request(self, method, url, headers=None, timeout=None, json=None, **kwargs):
        with self._lock:
            self.calls += 1
        text = answer_for(json['messages'][-1]['content'])

        class Response:
            status_code = 200
            headers = {}

            @staticmethod
            def json():
                return {'model': json['model'], 'provider': 'Novita',
                        'choices': [{'finish_reason': 'stop', 'message': {'content': text}}],
                        'usage': {'cost': 0.002}}

        return Response()


def test_cached_rerun_is_free(nlp_model):
    session = PromptAwareSession()
    client = OpenRouterClient(api_key='test-key', session=session, cache=ResponseCache())

    first = make_explainer(nlp_model, client).interpret(PROMPT, CLUSTERS)
    second = make_explainer(nlp_model, client).interpret(PROMPT, CLUSTERS)

    assert session.calls == 5
    assert first.total_cost == pytest.approx(0.01)
    assert second.total_cost == 0.0 and second.n_api_requests == 0
    assert second.importance_df.equals(first.importance_df)


def test_influence_matrix_renders_with_existing_heatmap(nlp_model, monkeypatch, capsys):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: None)

    out = make_explainer(nlp_model).interpret(PROMPT, CLUSTERS)
    out.show_cluster_imp_aggr_heatmap()
    plt.close('all')

    assert 'Данные отсутствуют' not in capsys.readouterr().out


# ---------------------------------------------------------------- logprobs

from tests.test_logprob_scoring import FakeTokenLM, ON_PATH, ALTERNATIVE, OFF_PATH  # noqa: E402
from explainitall.api_interp.openrouter_client import OpenRouterError  # noqa: E402

REFERENCE_LOGPROB = 5 * ON_PATH
NO_DISEASE_LOGPROB = ON_PATH + ALTERNATIVE + 3 * OFF_PATH
NO_ANIMAL_LOGPROB = ALTERNATIVE + 4 * OFF_PATH


def test_logprob_drop_with_semantic_generation(nlp_model):
    lm = FakeTokenLM()
    out = make_explainer(nlp_model, lm, logprobs=True).interpret(PROMPT, CLUSTERS)

    disease, animals = row(out.importance_df, 'Болезни'), row(out.importance_df, 'Животные')
    assert disease['logprob_drop_total'] == pytest.approx(REFERENCE_LOGPROB - NO_DISEASE_LOGPROB)
    assert disease['logprob_drop'] == pytest.approx((REFERENCE_LOGPROB - NO_DISEASE_LOGPROB) / 5)
    assert animals['logprob_drop'] == pytest.approx((REFERENCE_LOGPROB - NO_ANIMAL_LOGPROB) / 5)
    assert disease['logprob_significant'] and animals['logprob_significant']
    assert (disease['logprob_tokens'], disease['logprob_truncated'], disease['logprob_calls']) == (5, 0, 1)
    assert np.isnan(row(out.importance_df, 'Лекарства')['logprob_drop'])

    analysis = out.logprob_analysis
    assert analysis.reference_tokens == [' Кошке', ' при', ' гриппе', ' помогут', ' уколы']
    assert analysis.prefill.supported is True
    assert analysis.noise_max == 0.0
    assert out.warnings == []
    # 3 исходных + 2 без кластеров + 2 проверки prefill + по одному перезапуску на кластер
    assert len(lm.calls) == out.n_api_requests == 9
    assert out.total_cost == pytest.approx(0.009)
    assert disease['importance'] > 0  # семантическая оценка тоже посчитана


def test_logprobs_without_semantic_generation(nlp_model):
    lm = FakeTokenLM()
    explainer = ExplainerAPI(client=lm, model='m/x', nlp_model=nlp_model, semantic=False, logprobs=True)
    out = explainer.interpret(PROMPT, CLUSTERS)

    assert list(out.importance_df.columns) == ['cluster', 'words', 'logprob_drop', 'logprob_drop_total',
                                               'logprob_tokens', 'logprob_truncated', 'logprob_calls',
                                               'logprob_significant']
    assert list(out.importance_df['cluster']) == ['Животные', 'Болезни', 'Лекарства']
    assert out.cluster_imp_aggr_df.empty
    assert row(out.importance_df, 'Болезни')['logprob_calls'] == 2
    assert len(lm.calls) == 3 + 2 + 2 * 2


def test_prefill_not_supported_falls_back_to_prefix(nlp_model):
    lm = FakeTokenLM(prefill=False)
    out = make_explainer(nlp_model, lm, logprobs=True).interpret(PROMPT, CLUSTERS)

    disease, animals = row(out.importance_df, 'Болезни'), row(out.importance_df, 'Животные')
    assert out.logprob_analysis.prefill.supported is False
    assert len(out.warnings) == 1 and 'prefill' in out.warnings[0]
    assert disease['logprob_tokens'] == 2
    assert disease['logprob_drop'] == pytest.approx((2 * ON_PATH - (ON_PATH + ALTERNATIVE)) / 2)
    assert animals['logprob_tokens'] == 1
    assert disease['logprob_calls'] == animals['logprob_calls'] == 0
    assert len(lm.calls) == 3 + 2 + 1


def test_explicit_prefill_skips_verification(nlp_model):
    lm = FakeTokenLM()
    out = make_explainer(nlp_model, lm, logprobs=True, use_prefill=True).interpret(PROMPT, CLUSTERS)

    assert out.logprob_analysis.prefill is None
    assert len(lm.calls) == 3 + 2 + 2


def test_logprob_noise_from_baseline_repeats(nlp_model):
    lm = FakeTokenLM(jitter={1: -0.1})
    out = make_explainer(nlp_model, lm, logprobs=True).interpret(PROMPT, CLUSTERS)

    assert out.logprob_analysis.noise_max == pytest.approx(0.1)
    assert out.logprob_analysis.noise_mean == pytest.approx(0.05)
    assert row(out.importance_df, 'Болезни')['logprob_significant']


def test_truncation_warning(nlp_model):
    lm = FakeTokenLM()
    out = make_explainer(nlp_model, lm, logprobs=True, top_logprobs=1).interpret(PROMPT, CLUSTERS)

    assert row(out.importance_df, 'Болезни')['logprob_truncated'] == 1
    assert any('top_logprobs' in w for w in out.warnings)


def test_provider_without_logprobs_is_reported(nlp_model):
    with pytest.raises(OpenRouterError, match='logprobs'):
        make_explainer(nlp_model, FakeTokenLM(logprobs=False), logprobs=True).interpret(PROMPT, CLUSTERS)


def test_plan_request_bounds_with_logprobs(nlp_model):
    plan = make_explainer(nlp_model, logprobs=True, max_scoring_calls=8).plan(PROMPT, CLUSTERS)
    plan_no_semantic = ExplainerAPI(client=FakeTokenLM(), model='m/x', nlp_model=nlp_model, semantic=False,
                                    logprobs=True, use_prefill=False).plan(PROMPT, CLUSTERS)

    assert (plan.n_requests, plan.n_requests_max) == (5, 5 + 2 + 4 * 8)
    assert (plan_no_semantic.n_requests, plan_no_semantic.n_requests_max) == (5, 3 + 4 * 8)


@pytest.mark.parametrize('kwargs', [
    {'semantic': False, 'logprobs': False},
    {'semantic': True, 'embedder': None},
    {'logprobs': True, 'use_prefill': 'yes'},
])
def test_invalid_scoring_settings(nlp_model, kwargs):
    params = dict(client=FakeTokenLM(), model='m/x', nlp_model=nlp_model, embedder=BagOfWordsEmbedder())
    params.update(kwargs)

    with pytest.raises(ValueError):
        ExplainerAPI(**params)
