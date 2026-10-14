from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .openrouter_client import OpenRouterClient, ProviderSettings
from .perturbations import TextOcclusion


class RerankerExplainer:
    """
    Интерпретация реранкера OpenRouter (/rerank) исключением слов и кластеров.

    - document_*_importance: насколько падает релевантность документа без слова/кластера документа;
      все варианты документа оцениваются в одном запросе вместе с исходным, поэтому сравнение корректно
      и для моделей, нормирующих оценки внутри запроса;
    - query_*_importance: насколько меняется релевантность каждого документа без слова/кластера запроса
      и меняется ли самый релевантный документ (один запрос на каждый вариант запроса).
    importance > 0 — элемент повышает релевантность, < 0 — понижает.
    """

    def __init__(self, client: OpenRouterClient, model: str, provider: Optional[ProviderSettings] = None,
                 nlp_model=None, replacement: str = '', max_documents_per_request: int = 100):
        """
        nlp_model: векторная модель слов gensim; нужна для кластеров
        replacement: чем заменять скрытые слова; пустая строка — удалить
        max_documents_per_request: сколько документов отправлять в одном запросе
        """
        if max_documents_per_request < 2:
            raise ValueError('max_documents_per_request должен быть не меньше 2')
        self.client = client
        self.model = model
        self.provider = provider
        self.max_documents_per_request = max_documents_per_request
        self.occlusion = TextOcclusion(nlp_model, replacement)
        self.total_cost = 0.0
        self.n_requests = 0

    def _rerank(self, query: str, documents: List[str]) -> List[float]:
        result = self.client.rerank(query, documents, self.model, provider=self.provider)
        if not result.cached:
            self.total_cost += result.cost
            self.n_requests += 1
        return result.scores

    def scores(self, query: str, documents: Sequence[str]) -> List[float]:
        """Релевантность документов запросу в порядке documents"""
        documents = list(documents)
        scores = []
        for start in range(0, len(documents), self.max_documents_per_request):
            scores += self._rerank(query, documents[start:start + self.max_documents_per_request])
        return scores

    def _document_shift(self, query: str, document: str, groups):
        """(исходная релевантность, релевантность каждого варианта, падение относительно исходной в той же порции)"""
        variants = self.occlusion.variants(document, groups)
        chunk = self.max_documents_per_request - 1
        base_scores, without, drops = [], [], []
        for start in range(0, len(variants), chunk):
            part = variants[start:start + chunk]
            base, *scores = self._rerank(query, [document, *part])
            base_scores.append(base)
            without += scores
            drops += [base - s for s in scores]
        if not variants:
            base_scores.append(self._rerank(query, [document])[0])
        return float(np.mean(base_scores)), without, drops

    def document_word_importance(self, query: str, document: str) -> pd.DataFrame:
        words, groups = self.occlusion.word_groups(document)
        if not words:
            raise ValueError('В документе нет слов')
        base, without, drops = self._document_shift(query, document, groups)
        return pd.DataFrame({'word': words, 'start': [g[0].start for g in groups], 'relevance': base,
                             'relevance_without': without, 'importance': drops})

    def document_cluster_importance(self, query: str, document: str, clusters_description: List[dict]) -> pd.DataFrame:
        names, words, groups = self.occlusion.cluster_groups(document, clusters_description)
        if not names:
            raise ValueError('В документе не найдено ни одного слова из заданных кластеров')
        base, without, drops = self._document_shift(query, document, groups)
        df = pd.DataFrame({'cluster': names, 'words': words, 'relevance': base, 'relevance_without': without,
                           'importance': drops})
        return df.sort_values('importance', ascending=False, ignore_index=True)

    def _query_shift(self, query: str, documents: List[str], labels: List[str], words: List[List[str]], groups,
                     label_column: str) -> pd.DataFrame:
        base = self.scores(query, documents)
        base_top = int(np.nanargmax(base))
        rows = []
        for label, label_words, variant in zip(labels, words, self.occlusion.variants(query, groups)):
            scores = self.scores(variant, documents)
            row = {label_column: label, 'words': label_words}
            row.update({f'doc_{i}': b - s for i, (b, s) in enumerate(zip(base, scores))})
            row['mean_importance'] = float(np.nanmean([b - s for b, s in zip(base, scores)]))
            row['top1_without'] = int(np.nanargmax(scores))
            row['top1_changed'] = row['top1_without'] != base_top
            rows.append(row)
        return pd.DataFrame(rows)

    def query_word_importance(self, query: str, documents: Sequence[str]) -> pd.DataFrame:
        """Строки — слова запроса; doc_i — падение релевантности документа i без слова"""
        words, groups = self.occlusion.word_groups(query)
        if not words:
            raise ValueError('В запросе нет слов')
        return self._query_shift(query, list(documents), words, [[w] for w in words], groups, 'word')

    def query_cluster_importance(self, query: str, documents: Sequence[str],
                                 clusters_description: List[dict]) -> pd.DataFrame:
        """Строки — кластеры запроса; doc_i — падение релевантности документа i без кластера"""
        names, words, groups = self.occlusion.cluster_groups(query, clusters_description)
        if not names:
            raise ValueError('В запросе не найдено ни одного слова из заданных кластеров')
        df = self._query_shift(query, list(documents), names, words, groups, 'cluster')
        return df.sort_values('mean_importance', ascending=False, ignore_index=True)
