from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .openrouter_client import OpenRouterClient, ProviderSettings
from .perturbations import TextOcclusion


class APIEmbedder:
    """
    Эмбеддер OpenRouter (/embeddings) с методом encode, как у SentenceTransformer.
    Подходит везде, где библиотека ждёт эмбеддер: EmbedderExplainer, ExplainerAPI(embedder=...).
    """

    def __init__(self, client: OpenRouterClient, model: str, dimensions: Optional[int] = None,
                 input_type: Optional[str] = None, provider: Optional[ProviderSettings] = None,
                 batch_size: int = 64):
        """
        input_type: для асимметричных моделей — 'search_query' или 'search_document'
        batch_size: сколько текстов отправлять в одном запросе
        """
        self.client = client
        self.model = model
        self.dimensions = dimensions
        self.input_type = input_type
        self.provider = provider
        self.batch_size = batch_size
        self.total_cost = 0.0
        self.n_requests = 0

    def encode(self, texts: Union[str, Sequence[str]], **kwargs) -> np.ndarray:
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            result = self.client.embeddings(texts[start:start + self.batch_size], self.model,
                                            dimensions=self.dimensions, input_type=self.input_type,
                                            provider=self.provider)
            if not result.cached:
                self.total_cost += result.cost
                self.n_requests += 1
            vectors.append(result.vectors)
        matrix = np.vstack(vectors) if vectors else np.zeros((0, 0))
        return matrix[0] if single else matrix


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def _with_share(df: pd.DataFrame, column: str = 'importance') -> pd.DataFrame:
    positive = df[column].clip(lower=0)
    total = positive.sum()
    df['share'] = positive / total if total > 0 else 0.0
    return df


@dataclass
class SimilarityExplanation:
    similarity: float  # косинусная близость запроса и документа
    contributions: pd.DataFrame  # side, label, words, similarity_without, contribution

    def top(self, n: int = 10) -> pd.DataFrame:
        return self.contributions.sort_values('contribution', ascending=False).head(n)


class EmbedderExplainer:
    """
    Интерпретация эмбеддера (за API или локального) исключением слов и кластеров.

    - word_importance / cluster_importance: насколько эмбеддинг текста меняется без слова или кластера
      (1 - косинус между эмбеддингами исходного и изменённого текста);
    - similarity_attribution: какие слова запроса и документа обеспечивают их близость
      (contribution > 0 — без слова близость падает, < 0 — растёт).
    Эмбеддинги детерминированы, поэтому повторы для оценки шума не нужны; все варианты текста
    уходят в одном пакетном запросе.
    """

    def __init__(self, embedder, nlp_model=None, replacement: str = '', query_embedder=None):
        """
        embedder: объект с encode(list[str]) -> массив, например APIEmbedder или SentenceTransformer
        nlp_model: векторная модель слов gensim; нужна для кластеров
        replacement: чем заменять скрытые слова; пустая строка — удалить
        query_embedder: отдельный эмбеддер запросов для асимметричных моделей (input_type='search_query')
        """
        self.embedder = embedder
        self.query_embedder = query_embedder or embedder
        self.occlusion = TextOcclusion(nlp_model, replacement)

    def _shift(self, text: str, groups) -> np.ndarray:
        """1 - косинус между эмбеддингом текста и эмбеддингом каждого варианта без группы слов"""
        vectors = _normalize(self.embedder.encode([text, *self.occlusion.variants(text, groups)]))
        return 1.0 - vectors[1:] @ vectors[0]

    def word_importance(self, text: str) -> pd.DataFrame:
        words, groups = self.occlusion.word_groups(text)
        if not words:
            raise ValueError('В тексте нет слов')
        df = pd.DataFrame({'word': words, 'start': [g[0].start for g in groups],
                           'importance': self._shift(text, groups)})
        return _with_share(df)

    def cluster_importance(self, text: str, clusters_description: List[dict]) -> pd.DataFrame:
        names, words, groups = self.occlusion.cluster_groups(text, clusters_description)
        if not names:
            raise ValueError('В тексте не найдено ни одного слова из заданных кластеров')
        df = pd.DataFrame({'cluster': names, 'words': words, 'importance': self._shift(text, groups)})
        return _with_share(df).sort_values('importance', ascending=False, ignore_index=True)

    def _similarity_explanation(self, query: str, document: str, side: str, query_groups, document_groups,
                                query_labels, document_labels) -> SimilarityExplanation:
        query_variants = self.occlusion.variants(query, query_groups) if side in ('query', 'both') else []
        document_variants = self.occlusion.variants(document, document_groups) if side in ('document', 'both') else []

        queries = _normalize(self.query_embedder.encode([query, *query_variants]))
        documents = _normalize(self.embedder.encode([document, *document_variants]))
        similarity = float(queries[0] @ documents[0])

        rows = []
        for (label, words), sim in zip(query_labels, queries[1:] @ documents[0]):
            rows.append({'side': 'query', 'label': label, 'words': words, 'similarity_without': float(sim)})
        for (label, words), sim in zip(document_labels, documents[1:] @ queries[0]):
            rows.append({'side': 'document', 'label': label, 'words': words, 'similarity_without': float(sim)})
        df = pd.DataFrame(rows, columns=['side', 'label', 'words', 'similarity_without'])
        df['contribution'] = similarity - df['similarity_without']
        return SimilarityExplanation(similarity=similarity, contributions=df)

    def similarity_attribution(self, query: str, document: str, side: str = 'both') -> SimilarityExplanation:
        """Вклад каждого слова запроса и/или документа в их близость; side: 'query', 'document' или 'both'"""
        if side not in ('query', 'document', 'both'):
            raise ValueError("side должен быть 'query', 'document' или 'both'")
        query_words, query_groups = self.occlusion.word_groups(query)
        document_words, document_groups = self.occlusion.word_groups(document)
        return self._similarity_explanation(query, document, side, query_groups, document_groups,
                                            [(w, [w]) for w in query_words], [(w, [w]) for w in document_words])

    def cluster_similarity_attribution(self, query: str, document: str, clusters_description: List[dict],
                                       side: str = 'both') -> SimilarityExplanation:
        """Вклад кластеров запроса и/или документа в их близость"""
        if side not in ('query', 'document', 'both'):
            raise ValueError("side должен быть 'query', 'document' или 'both'")
        q_names, q_words, q_groups = self.occlusion.cluster_groups(query, clusters_description)
        d_names, d_words, d_groups = self.occlusion.cluster_groups(document, clusters_description)
        return self._similarity_explanation(query, document, side, q_groups, d_groups,
                                            list(zip(q_names, q_words)), list(zip(d_names, d_words)))
