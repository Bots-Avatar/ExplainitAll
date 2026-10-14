from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from explainitall.gpt_like_interp.interp import ExplainerGPT2, ExplainerGPT2Output
from .capabilities import get_models
from .openrouter_client import OpenRouterClient, ProviderSettings
from .perturbations import Messages, UNNAMED_CLUSTER, normalize_messages


def _tokenizer_adds_bos(tokenizer) -> bool:
    if tokenizer.bos_token_id is None:
        return False
    ids = tokenizer('a')['input_ids']
    return bool(ids) and ids[0] == tokenizer.bos_token_id


def chat_prompt_text(tokenizer, messages: Messages) -> str:
    """
    Текст промпта в том виде, в каком его видит модель за API: с шаблоном чата и приглашением ассистента.
    BOS из шаблона убирается, если токенизатор добавляет его сам, иначе он оказался бы в начале дважды.
    """
    messages = normalize_messages(messages)
    if getattr(tokenizer, 'chat_template', None):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = '\n'.join(m['content'] for m in messages) + '\n'
    bos = tokenizer.bos_token
    if bos and text.startswith(bos) and _tokenizer_adds_bos(tokenizer):
        text = text[len(bos):]
    return text


def resolve_hf_model(client: OpenRouterClient, model: str) -> Optional[str]:
    """Репозиторий Hugging Face с открытыми весами модели OpenRouter (поле hugging_face_id), если он указан"""
    for info in get_models(client):
        if info.id == model:
            return info.hugging_face_id
    return None


def input_cluster_importance(explanation: ExplainerGPT2Output, aggr: str = 'mean') -> pd.Series:
    """
    Важность кластеров промпта по атрибуциям: средняя по сгенерированным словам нормированная
    атрибуция слов промпта, агрегированная по кластерам. Слова вне кластеров не учитываются.
    """
    n_input = len(explanation.attributions_grouped_norm.tokens_input)
    df = explanation.cluster_imp_df.iloc[:n_input]
    values = df.drop(columns='Tokens').mean(axis=1)
    importance = values.groupby(df['Tokens'].values).agg(aggr)
    return importance.drop(index=UNNAMED_CLUSTER, errors='ignore').sort_values(ascending=False)


@dataclass
class SurrogateOutput:
    api_answer: str
    prompt_text: str
    hf_model: Optional[str]
    explanation: ExplainerGPT2Output
    api_cost: float

    def input_cluster_importance(self, aggr: str = 'mean') -> pd.Series:
        return input_cluster_importance(self.explanation, aggr)


class SurrogateExplainer:
    """
    Интерпретация модели за API через её открытого двойника.

    Ответ берётся у модели за API, а атрибуции (Integrated Gradients) считаются локально
    на открытых весах той же модели по тому же тексту промпта с шаблоном чата.
    Провайдер может использовать квантованные веса (fp8, fp4), поэтому результат двойника
    стоит сверять с оценками чёрного ящика (ExplainerAPI, agreement.rank_agreement).
    """

    def __init__(self,
                 client: OpenRouterClient,
                 model: str,
                 nlp_model,
                 attribution_model=None,
                 hf_model: Optional[str] = None,
                 provider: Optional[ProviderSettings] = None,
                 max_tokens: Optional[int] = 256,
                 temperature: Optional[float] = 0.0,
                 seed: Optional[int] = None,
                 answer_max_tokens: Optional[int] = 64,
                 steps: int = 34,
                 batch_size: int = 50,
                 attribution_method: str = 'integrated_gradients'):
        """
        attribution_model: готовая модель inseq; если не задана, загружается по hf_model
        hf_model: репозиторий открытых весов; по умолчанию берётся hugging_face_id модели в OpenRouter
        answer_max_tokens: сколько первых токенов ответа объяснять (время IG растёт с длиной ответа)
        """
        self.client = client
        self.model = model
        self.nlp_model = nlp_model
        self.attribution_model = attribution_model
        self.hf_model = hf_model
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.answer_max_tokens = answer_max_tokens
        self.steps = steps
        self.batch_size = batch_size
        self.attribution_method = attribution_method

    def get_attribution_model(self):
        if self.attribution_model is None:
            if self.hf_model is None:
                self.hf_model = resolve_hf_model(self.client, self.model)
            if not self.hf_model:
                raise ValueError(f'Для {self.model} не найдены открытые веса: передайте hf_model или attribution_model')
            from inseq import load_model
            self.attribution_model = load_model(self.hf_model, attribution_method=self.attribution_method)
        return self.attribution_model

    def truncate_answer(self, tokenizer, answer: str) -> str:
        if self.answer_max_tokens is None:
            return answer
        ids = tokenizer(answer, add_special_tokens=False)['input_ids']
        if len(ids) <= self.answer_max_tokens:
            return answer
        return tokenizer.decode(ids[:self.answer_max_tokens], skip_special_tokens=True)

    def interpret(self, prompt: Messages, clusters_description: List[dict],
                  answer: Optional[str] = None) -> SurrogateOutput:
        """answer: уже полученный ответ модели за API; если не задан, запрашивается"""
        messages = normalize_messages(prompt)
        api_cost = 0.0
        if answer is None:
            result = self.client.chat(messages, self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                                      seed=self.seed, provider=self.provider)
            answer = result.text
            api_cost = 0.0 if result.cached else result.cost

        attribution_model = self.get_attribution_model()
        tokenizer = attribution_model.tokenizer
        prompt_text = chat_prompt_text(tokenizer, messages)
        explained_answer = self.truncate_answer(tokenizer, answer)
        if not explained_answer.strip():
            raise ValueError('Модель за API вернула пустой ответ: объяснять нечего')

        explainer = ExplainerGPT2(gpt_model=attribution_model, nlp_model=self.nlp_model)
        explanation = explainer.interpret(input_texts=prompt_text, generated_texts=explained_answer,
                                          clusters_description=clusters_description,
                                          batch_size=self.batch_size, steps=self.steps)
        return SurrogateOutput(api_answer=answer, prompt_text=prompt_text, hf_model=self.hf_model,
                               explanation=explanation, api_cost=api_cost)
