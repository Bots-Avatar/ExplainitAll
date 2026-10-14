import numpy as np
import pandas as pd
import pytest

inseq = pytest.importorskip("inseq")
gensim = pytest.importorskip("gensim")
pytest.importorskip("pymorphy3")

from explainitall.api_interp.agreement import api_cluster_importance, rank_agreement
from explainitall.api_interp.openrouter_client import ChatResult
from explainitall.api_interp.surrogate import (SurrogateExplainer, chat_prompt_text, input_cluster_importance,
                                               resolve_hf_model)
from explainitall.gpt_like_interp.inseq_helpers import AttrObj
from explainitall.gpt_like_interp.interp import ExplainerGPT2Output

PROMPT = 'У моей кошки грипп и чуть-чуть температура. Чем лечить кошку?'
ANSWER = 'Кошке при гриппе помогут уколы.'
CLUSTERS = [{'name': 'Животные', 'centroid': ['кошка'], 'top_k': 1},
            {'name': 'Болезни', 'centroid': ['грипп'], 'top_k': 1},
            {'name': 'Лекарства', 'centroid': ['укол'], 'top_k': 1}]


class FakeTokenizer:
    def __init__(self, chat_template=None, bos_token=None, bos_token_id=None, adds_bos=False):
        self.chat_template = chat_template
        self.bos_token = bos_token
        self.bos_token_id = bos_token_id
        self.adds_bos = adds_bos

    def __call__(self, text, **kwargs):
        return {'input_ids': ([self.bos_token_id] if self.adds_bos else []) + [7]}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        body = ''.join(f"<{m['role']}>{m['content']}" for m in messages)
        return (self.bos_token or '') + body + ('<assistant>' if add_generation_prompt else '')


class FakeAPIClient:
    def __init__(self, answer=ANSWER, models=None):
        self.answer = answer
        self.models = models or []
        self.calls = []

    def chat(self, messages, model, **kwargs):
        self.calls.append(messages)
        return ChatResult(text=self.answer, finish_reason='stop', model=model, provider='Novita', usage={},
                          cost=0.001, logprobs=None, reasoning=None, cached=False, raw={})

    def list_models(self, supported_parameters=None):
        return self.models


def test_chat_prompt_text_uses_template():
    text = chat_prompt_text(FakeTokenizer(chat_template='x'), 'Вопрос?')

    assert text == '<user>Вопрос?<assistant>'


def test_chat_prompt_text_without_template():
    messages = [{'role': 'system', 'content': 'Ты врач.'}, {'role': 'user', 'content': 'Вопрос?'}]

    assert chat_prompt_text(FakeTokenizer(), messages) == 'Ты врач.\nВопрос?\n'


@pytest.mark.parametrize('adds_bos,expected', [(True, '<user>Q<assistant>'), (False, '<s><user>Q<assistant>')])
def test_chat_prompt_text_avoids_double_bos(adds_bos, expected):
    tokenizer = FakeTokenizer(chat_template='x', bos_token='<s>', bos_token_id=1, adds_bos=adds_bos)

    assert chat_prompt_text(tokenizer, 'Q') == expected


def test_resolve_hf_model():
    client = FakeAPIClient(models=[{'id': 'meta-llama/llama-3.1-8b-instruct',
                                    'hugging_face_id': 'meta-llama/Meta-Llama-3.1-8B-Instruct'},
                                   {'id': 'anthropic/claude', 'hugging_face_id': ''}])

    assert resolve_hf_model(client, 'meta-llama/llama-3.1-8b-instruct') == 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    assert resolve_hf_model(client, 'anthropic/claude') is None
    assert resolve_hf_model(client, 'unknown/model') is None


def test_closed_model_without_weights_is_reported():
    explainer = SurrogateExplainer(FakeAPIClient(models=[{'id': 'anthropic/claude', 'hugging_face_id': ''}]),
                                   'anthropic/claude', nlp_model=None)

    with pytest.raises(ValueError, match='открытые веса'):
        explainer.get_attribution_model()


def test_input_cluster_importance_uses_only_prompt_rows():
    attr = AttrObj(phrase_input='', phrase_generated='', tokens_input=('у', 'кошки', 'грипп', 'кошку'),
                   tokens_generated=('уколы', 'помогут'), array=np.zeros((6, 2)))
    cluster_df = pd.DataFrame({'Tokens': ['unnamed', 'Животные', 'Болезни', 'Животные', 'Лекарства', 'unnamed'],
                               'Лекарства': [0.9, 0.2, 0.8, 0.4, 0.99, 0.5],
                               'unnamed': [0.9, 0.4, 0.6, 0.2, 0.99, 0.5]})
    explanation = ExplainerGPT2Output(attributions=attr, attributions_grouped=attr, attributions_grouped_norm=attr,
                                      cluster_imp_df=cluster_df, cluster_imp_aggr_df=pd.DataFrame(),
                                      word_imp_df=pd.DataFrame(), word_imp_norm_df=pd.DataFrame())

    importance = input_cluster_importance(explanation)

    assert importance.to_dict() == pytest.approx({'Болезни': 0.7, 'Животные': 0.3})
    assert input_cluster_importance(explanation, aggr='max')['Животные'] == pytest.approx(0.3)


def test_rank_agreement():
    a = {'Болезни': 0.9, 'Животные': 0.5, 'Лекарства': 0.1}

    assert rank_agreement(a, {'Болезни': 3, 'Животные': 2, 'Лекарства': 1}).spearman == pytest.approx(1.0)
    assert rank_agreement(a, {'Болезни': 1, 'Животные': 2, 'Лекарства': 3}).spearman == pytest.approx(-1.0)

    partial = rank_agreement(a, {'Болезни': 0.4, 'Животные': 0.6, 'Симптомы': 1.0})
    assert partial.n_common == 2 and np.isnan(partial.spearman) and not partial.top1_match
    assert rank_agreement(a, {'Другое': 1.0}).n_common == 0


def test_api_cluster_importance_drops_missing_clusters():
    df = pd.DataFrame({'cluster': ['Болезни', 'Животные', 'Лекарства'], 'importance': [0.7, 0.2, np.nan]})

    assert api_cluster_importance(df).to_dict() == {'Болезни': 0.7, 'Животные': 0.2}


# ------------------------------------------------------------ интеграция

@pytest.fixture(scope="module")
def nlp_model():
    words = {'кошка_NOUN': [1, 0, 0, 0], 'собака_NOUN': [0.9, 0.1, 0, 0], 'грипп_NOUN': [0, 1, 0, 0],
             'простуда_NOUN': [0, 0.9, 0.1, 0], 'укол_NOUN': [0, 0, 1, 0], 'таблетка_NOUN': [0, 0, 0.9, 0.1]}
    kv = gensim.models.KeyedVectors(vector_size=4)
    kv.add_vectors(list(words), np.array(list(words.values()), dtype=np.float32))
    return kv


def tiny_model(family, tmp_path_factory):
    """Крошечная случайная модель семейства с настоящим токенизатором: скачивается только токенизатор"""
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    tokenizer_repo, config_cls, model_cls = {
        'qwen2': ('Qwen/Qwen2.5-0.5B-Instruct', Qwen2Config, Qwen2ForCausalLM),
        'llama': ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', LlamaConfig, LlamaForCausalLM),
    }[family]
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
    except OSError as e:
        pytest.skip(f'Не удалось загрузить токенизатор {tokenizer_repo}: {e}')

    torch.manual_seed(0)
    # inseq берёт bos_token_id из конфига модели; у Qwen BOS нет, и, как в настоящем конфиге, им служит <|endoftext|>
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    config = config_cls(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                        bos_token_id=bos_token_id, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id)
    model_dir = tmp_path_factory.mktemp(f'tiny_{family}')
    model_cls(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return inseq.load_model(str(model_dir), attribution_method='integrated_gradients')


@pytest.mark.parametrize('family,template_marker', [('qwen2', '<|im_start|>assistant'), ('llama', '<|assistant|>')])
def test_surrogate_explains_api_answer(family, template_marker, nlp_model, tmp_path_factory):
    client = FakeAPIClient()
    explainer = SurrogateExplainer(client, 'org/model', nlp_model=nlp_model,
                                   attribution_model=tiny_model(family, tmp_path_factory), steps=4, batch_size=8)

    out = explainer.interpret(PROMPT, CLUSTERS)

    assert len(client.calls) == 1
    assert out.api_answer == ANSWER and out.api_cost == pytest.approx(0.001)
    assert out.prompt_text.endswith(template_marker + '\n') or template_marker in out.prompt_text[-40:]
    words = list(out.explanation.word_imp_df['Tokens'])
    assert {'грипп', 'чуть-чуть', 'уколы'} <= set(words)

    importance = out.input_cluster_importance()
    assert {'Животные', 'Болезни'} <= set(importance.index)
    assert 'Лекарства' not in importance.index  # в промпте нет слов этого кластера


def test_surrogate_uses_given_answer_and_truncates(nlp_model, tmp_path_factory):
    client = FakeAPIClient()
    explainer = SurrogateExplainer(client, 'org/model', nlp_model=nlp_model,
                                   attribution_model=tiny_model('qwen2', tmp_path_factory), steps=4, batch_size=8,
                                   answer_max_tokens=3)

    out = explainer.interpret(PROMPT, CLUSTERS, answer=ANSWER)

    assert client.calls == [] and out.api_cost == 0.0
    assert len(out.explanation.attributions.tokens_generated) == 3
