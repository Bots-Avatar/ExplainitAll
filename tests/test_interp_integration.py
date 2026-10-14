import numpy as np
import pytest

inseq = pytest.importorskip("inseq")
gensim = pytest.importorskip("gensim")
pytest.importorskip("pymorphy3")

from explainitall.gpt_like_interp import interp

RU_TOKENIZER = "ai-forever/rugpt3small_based_on_gpt2"


@pytest.fixture(scope="module")
def gpt_model(tmp_path_factory):
    """Крошечная случайная GPT-2 с русским BPE-токенизатором ruGPT-3: скачивается только токенизатор"""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(RU_TOKENIZER)
    except OSError as e:
        pytest.skip(f"Не удалось загрузить токенизатор {RU_TOKENIZER}: {e}")

    torch.manual_seed(0)
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=128, n_embd=16, n_layer=2, n_head=2)
    model_dir = tmp_path_factory.mktemp("tiny_ru_gpt2")
    GPT2LMHeadModel(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return inseq.load_model(model=str(model_dir), attribution_method="integrated_gradients")


@pytest.fixture(scope="module")
def nlp_model():
    words = ['кошка_NOUN', 'собака_NOUN', 'грипп_NOUN', 'простуда_NOUN',
             'укол_NOUN', 'таблетка_NOUN', 'лечить_VERB']
    rng = np.random.default_rng(0)
    kv = gensim.models.KeyedVectors(vector_size=8)
    kv.add_vectors(words, rng.normal(size=(len(words), 8)).astype(np.float32))
    return kv


def test_interpret_with_hyphenated_words(gpt_model, nlp_model):
    explainer = interp.ExplainerGPT2(gpt_model=gpt_model, nlp_model=nlp_model)
    clusters = [{'name': 'Животные', 'centroid': ['кошка'], 'top_k': 1},
                {'name': 'Болезни', 'centroid': ['грипп'], 'top_k': 1},
                {'name': 'Лекарства', 'centroid': ['укол'], 'top_k': 1}]

    out = explainer.interpret(input_texts='у кошки грипп и чуть-чуть температура: чем лечить кошку? ответ:',
                              generated_texts=' лечите её уколами',
                              clusters_description=clusters,
                              batch_size=4,
                              steps=4)

    words = list(out.word_imp_df['Tokens'])
    assert 'чуть-чуть' in words
    assert 'грипп' in words
    assert all(isinstance(w, str) for w in words)
    assert out.word_imp_df.shape == out.word_imp_norm_df.shape
    assert out.cluster_imp_df.shape == out.word_imp_df.shape
    assert out.attributions_grouped.array.shape == (len(out.attributions_grouped.tokens_input) +
                                                    len(out.attributions_grouped.tokens_generated),
                                                    len(out.attributions_grouped.tokens_generated))

    cluster_rows = set(out.cluster_imp_df['Tokens'])
    assert {'Животные', 'unnamed'} <= cluster_rows
    assert not out.cluster_imp_aggr_df.empty
    assert set(out.cluster_imp_aggr_df['Tokens']) <= {'Животные', 'Болезни', 'Лекарства'}
