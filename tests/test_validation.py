import numpy as np
import pandas as pd
import pytest

gensim = pytest.importorskip("gensim")
pytest.importorskip("pymorphy3")

from explainitall.api_interp.explainer import ExplainerAPI
from explainitall.api_interp.validation import (PromptValidation, ValidationExperiment, aopc, deletion_curve,
                                                ordering_from_importance, random_orderings, summarize)
from explainitall.api_interp.agreement import RankAgreement
from tests.test_api_explainer import BagOfWordsEmbedder, CLUSTERS, FakeChatClient, PROMPT
from tests.test_logprob_scoring import ALTERNATIVE, FakeTokenLM, OFF_PATH, ON_PATH

DROP_NO_ANIMAL = (5 * ON_PATH - (ALTERNATIVE + 4 * OFF_PATH)) / 5
DROP_NO_DISEASE = (5 * ON_PATH - (ON_PATH + ALTERNATIVE + 3 * OFF_PATH)) / 5


@pytest.fixture(scope="module")
def nlp_model():
    words = {'кошка_NOUN': [1, 0, 0, 0], 'собака_NOUN': [0.9, 0.1, 0, 0], 'грипп_NOUN': [0, 1, 0, 0],
             'простуда_NOUN': [0, 0.9, 0.1, 0], 'укол_NOUN': [0, 0, 1, 0], 'таблетка_NOUN': [0, 0, 0.9, 0.1]}
    kv = gensim.models.KeyedVectors(vector_size=4)
    kv.add_vectors(list(words), np.array(list(words.values()), dtype=np.float32))
    return kv


class StubTwin:
    def __init__(self, importance):
        self.importance = pd.Series(importance)
        self.answers = []

    def interpret(self, prompt, clusters_description, answer=None):
        self.answers.append(answer)
        twin = self

        class Output:
            @staticmethod
            def input_cluster_importance():
                return twin.importance

        return Output()


def make_explainer(nlp_model, client, **kwargs):
    params = dict(client=client, model='m/x', nlp_model=nlp_model, embedder=BagOfWordsEmbedder(), logprobs=True)
    params.update(kwargs)
    return ExplainerAPI(**params)


def test_masker_hides_several_clusters(nlp_model):
    masker = make_explainer(nlp_model, FakeTokenLM())._masker(CLUSTERS)

    masked = masker.mask_clusters(PROMPT, ['Животные', 'Болезни'])

    assert masked[0]['content'] == 'У моей [...] [...]. Чем лечить [...]?'
    with pytest.raises(KeyError):
        masker.mask_clusters(PROMPT, ['Нет такого'])


def test_ordering_from_importance_puts_unscored_last():
    importance = pd.Series({'Б': 0.1, 'А': 0.9, 'Лишний': 5.0})

    assert ordering_from_importance(importance, ['А', 'Б', 'В']) == ['А', 'Б', 'В']


def test_aopc_and_random_orderings():
    assert aopc([0.0, 0.5, 1.0]) == pytest.approx(0.75)
    assert np.isnan(aopc([0.0]))
    assert np.isnan(aopc([0.0, np.nan]))
    assert random_orderings(['a', 'b', 'c'], 3, seed=1) == random_orderings(['a', 'b', 'c'], 3, seed=1)
    assert all(sorted(o) == ['a', 'b', 'c'] for o in random_orderings(['a', 'b', 'c'], 4))


def test_deletion_curve_reuses_measurements(nlp_model):
    masker = make_explainer(nlp_model, FakeTokenLM())._masker(CLUSTERS)
    measured = []

    def measure(messages):
        measured.append(messages[0]['content'])
        return float(messages[0]['content'].count('[...]'))

    cache = {}
    first = deletion_curve(masker, PROMPT, ['Животные', 'Болезни'], measure, cache)
    second = deletion_curve(masker, PROMPT, ['Болезни', 'Животные'], measure, cache)

    assert first == [0.0, 2.0, 3.0]
    assert second == [0.0, 1.0, 3.0]
    assert len(measured) == 3  # {Ж}, {Ж, Б}, {Б}: полный набор посчитан один раз


def test_experiment_with_logprob_curves(nlp_model):
    lm = FakeTokenLM()
    twin = StubTwin({'Болезни': 0.7, 'Животные': 0.3})
    experiment = ValidationExperiment(make_explainer(nlp_model, lm), twin=twin, n_random=4)

    result = experiment.run_one(PROMPT, CLUSTERS)

    assert set(result.clusters) == {'Животные', 'Болезни'}
    assert twin.answers == [result.answer] and result.answer == ' Кошке при гриппе помогут уколы'
    assert set(result.rankings) == {'occlusion_semantic', 'occlusion_logprob', 'twin_ig'}

    # без животных — ответ рушится сильнее; без обоих кластеров модель отвечает как без животных
    assert result.curves['occlusion_logprob'] == pytest.approx([0.0, DROP_NO_ANIMAL, DROP_NO_ANIMAL])
    assert result.curves['twin_ig'] == pytest.approx([0.0, DROP_NO_DISEASE, DROP_NO_ANIMAL])
    assert result.aopc['occlusion_logprob'] > result.aopc['twin_ig']
    assert len(result.random_aopc) == 4

    assert result.agreements[('occlusion_semantic', 'twin_ig')].top1_match
    assert not result.agreements[('occlusion_logprob', 'twin_ig')].top1_match
    assert result.cost > 0


def test_experiment_with_semantic_curves(nlp_model):
    client = FakeChatClient()
    explainer = make_explainer(nlp_model, client, logprobs=False)
    result = ValidationExperiment(explainer, curve_measure='semantic', n_random=2).run_one(PROMPT, CLUSTERS)

    assert set(result.rankings) == {'occlusion_semantic'}
    curve = result.curves['occlusion_semantic']
    assert curve[0] == 0.0 and curve[1] > 0 and len(curve) == 3


@pytest.mark.parametrize('kwargs,measure', [({'logprobs': False}, 'logprob'),
                                            ({'semantic': False, 'embedder': None}, 'semantic'),
                                            ({}, 'other')])
def test_invalid_experiment_settings(nlp_model, kwargs, measure):
    with pytest.raises(ValueError):
        ValidationExperiment(make_explainer(nlp_model, FakeTokenLM(), **kwargs), curve_measure=measure)


def test_run_skips_prompts_without_cluster_words(nlp_model):
    experiment = ValidationExperiment(make_explainer(nlp_model, FakeTokenLM()), n_random=1)
    dataset = [(PROMPT, CLUSTERS), ('Какая сегодня погода?', CLUSTERS)]

    assert len(experiment.run(dataset, on_error='skip')) == 1
    with pytest.raises(ValueError):
        experiment.run(dataset)


def validation(aopc_values, random_values, clusters=('А', 'Б'), agreements=None):
    return PromptValidation(prompt=[], clusters=list(clusters), answer='', rankings={}, agreements=agreements or {},
                            curves={}, aopc=aopc_values, random_aopc=random_values, cost=0.0)


def test_summarize():
    results = [
        validation({'m1': 0.9, 'm2': 0.4}, [0.5, 0.7],
                   agreements={('m1', 'm2'): RankAgreement(spearman=1.0, n_common=3, top1_match=True)}),
        validation({'m1': 0.8, 'm2': 0.8}, [0.6],
                   agreements={('m1', 'm2'): RankAgreement(spearman=np.nan, n_common=2, top1_match=False)}),
        validation({'m1': 0.1}, [0.9], clusters=('А',)),  # один кластер: в кривых не учитывается
    ]

    methods, agreements = summarize(results)
    m1 = methods.set_index('method').loc['m1']

    assert m1['n_prompts'] == 2
    assert m1['aopc_mean'] == pytest.approx(0.85)
    assert m1['gain_over_random'] == pytest.approx(((0.9 - 0.6) + (0.8 - 0.6)) / 2)
    assert m1['better_than_random_share'] == 1.0
    assert methods.set_index('method').loc['m2', 'better_than_random_share'] == 0.5
    row = agreements.iloc[0]
    assert (row['n_prompts'], row['spearman_mean'], row['top1_match_share']) == (2, 1.0, 0.5)


# ---------------------------------------------------------------- local client

@pytest.fixture(scope="module")
def tiny_local_client(tmp_path_factory):
    pytest.importorskip("transformers")
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from explainitall.api_interp.local_client import LocalChatClient

    try:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    except OSError as e:
        pytest.skip(f'Не удалось загрузить токенизатор: {e}')
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                         bos_token_id=tokenizer.convert_tokens_to_ids('<|endoftext|>'),
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model = Qwen2ForCausalLM(config)
    model.generation_config.repetition_penalty = 1.5  # должен игнорироваться при расчёте logprobs
    return LocalChatClient(model, tokenizer, name='tiny-qwen')


def test_local_client_logprobs_match_forward_pass(tiny_local_client):
    import torch
    from explainitall.api_interp.surrogate import chat_prompt_text

    client = tiny_local_client
    result = client.chat('У кошки грипп. Чем лечить?', max_tokens=5, top_logprobs=3)

    assert len(result.logprobs) <= 5 and result.cost == 0.0 and result.provider == 'local'
    assert all(len(s.top_logprobs) == 3 for s in result.logprobs)
    assert all(s.top_logprobs[0][1] >= s.top_logprobs[-1][1] for s in result.logprobs)

    prompt_ids = client.tokenizer(chat_prompt_text(client.tokenizer, 'У кошки грипп. Чем лечить?'),
                                  return_tensors='pt')['input_ids']
    generated = client.tokenizer(''.join(s.token for s in result.logprobs), add_special_tokens=False,
                                 return_tensors='pt')['input_ids']
    with torch.no_grad():
        logits = client.model(torch.cat([prompt_ids, generated], dim=1)).logits[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    first = float(log_probs[prompt_ids.shape[1] - 1, generated[0, 0]])
    assert result.logprobs[0].logprob == pytest.approx(first, abs=1e-4)


def test_local_client_without_logprobs(tiny_local_client):
    result = tiny_local_client.chat('Привет', max_tokens=3)

    assert result.logprobs is None
    assert result.finish_reason in ('stop', 'length')
