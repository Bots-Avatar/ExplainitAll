import copy
from typing import List, Callable, Dict, Union, Optional

import gensim
from gensim.models import KeyedVectors
from pandas import DataFrame

from explainitall.gpt_like_interp.inseq_helpers import AttrObj
from . import nlp
from .gpt_like_interp import viz, inseq_helpers


class Cluster:
    def __init__(self, name: str,
                 sim_thresh: float,
                 words: list, matching_func: Callable):
        self.name = name  # Имя кластера
        self.sim_thresh = sim_thresh  # Порог схожести
        self.words = words  # Слова
        self.matching_func = matching_func  # Функция сравнения


class ClusterBuilder:
    def __init__(self, name: str, seed_words: list,
                 embeddings: KeyedVectors,
                 word_processor,
                 num_similar_words: int = 200,
                 pos_tags: Union[None, str, List[str]] = None):
        """
        name: Имя кластера
        seed_words: Начальный список слов
        embeddings: векторные представления слов
        num_similar_words: Количество наиболее похожих слов для поиска
        pos_tags: Части речи (UPOS) похожих слов, попадающих в кластер:
            None — те же, что у начальных слов; 'all' — любые; список, например ['NOUN', 'VERB'] — заданные.
            Начальные слова попадают в кластер при любой части речи.
        """
        self.name = name
        self.embeddings = embeddings
        self.num_similar_words = num_similar_words
        self.word_processor = word_processor
        self.pos_tags = pos_tags

        self.seed_words = seed_words

        self.embeddable_seed_words = self.word_processor.get_embeddable_words_batch(self.seed_words)

    def build(self, matching_func: Callable) -> Cluster:
        """
        Создает и возвращает объект Cluster
        matching_func: Функция для сравнения именованной сущности со словом
        """
        return Cluster(
            name=self.name,
            sim_thresh=0,
            words=self.find_similar_words(),
            matching_func=matching_func
        )

    @staticmethod
    def split_pos_tag(word_tagged: str):
        """'кошка_NOUN' -> ('кошка', 'NOUN')"""
        word, _, pos = word_tagged.rpartition("_")
        return word, pos

    def get_embeddable_word_from_most_similar(self, value) -> str:
        word_and_postfix, likelihood = value
        word, _ = self.split_pos_tag(word_and_postfix)
        return self.word_processor.get_embeddable_word_or_none(word)

    def get_allowed_pos_tags(self) -> Optional[set]:
        """Допустимые части речи похожих слов; None — без ограничений"""
        if self.pos_tags is None:
            return {self.split_pos_tag(w)[1] for w in self.embeddable_seed_words}
        if isinstance(self.pos_tags, str):
            return None if self.pos_tags.lower() == 'all' else {self.pos_tags}
        return set(self.pos_tags)

    def find_similar_words(self) -> list:
        if not self.embeddable_seed_words:
            return []

        try:
            similar_words = self.embeddings.most_similar(
                positive=self.embeddable_seed_words,
                topn=self.num_similar_words
            )
        except KeyError as e:
            raise Exception("embeddable_seed_words должны быть вида ['cat_NOUN','run_VERB']") from e

        neighbours = [self.get_embeddable_word_from_most_similar(result) for result in similar_words]
        neighbours = [w for w in neighbours if w]

        allowed_pos_tags = self.get_allowed_pos_tags()
        if allowed_pos_tags is not None:
            neighbours = [w for w in neighbours if self.split_pos_tag(w)[1] in allowed_pos_tags]

        words = [self.split_pos_tag(w)[0] for w in self.embeddable_seed_words + neighbours]
        return list(dict.fromkeys(words))


class ClusterManager:
    def __init__(self, embeddings: gensim.models.keyedvectors.KeyedVectors):
        self.embeddings = embeddings
        self.word_processor = nlp.WordProcessor(embeddings)

    def _is_same_normalized_word(self, word1: str, word2: str) -> bool:
        """
        Проверяет, одинаковы ли нормализованные формы двух слов.
        """
        return self.word_processor.get_normal_form_or_none(word1) == self.word_processor.get_normal_form_or_none(word2)

    def find_cluster_name(self, word: str, clusters: List[Cluster]) -> str:
        """
        Преобразует слово в имя кластера.
        """
        normalized_word = self.word_processor.get_normal_form_or_none(word)
        if normalized_word:
            for cluster in clusters:
                if normalized_word in cluster.words:
                    return cluster.name
        return "unnamed"

    def create_clusters(self, clusters_descr: List[Dict]):
        """
        Создает кластеры на основе их описаний.
        """
        clusters = []

        for descr in clusters_descr:
            clusters.append(ClusterBuilder(
                name=descr['name'],
                seed_words=descr['centroid'],
                embeddings=self.embeddings,
                num_similar_words=descr['top_k'],
                word_processor=self.word_processor,
                pos_tags=descr.get('pos')
            ).build(matching_func=self._is_same_normalized_word))
        return clusters


class ClusterInterpreter:
    """Интерпретация Кластеров"""

    def __init__(self,
                 clusters_discr: List[Dict[str, object]],
                 cluster_manager: ClusterManager
                 ):
        self.cluster_manager = cluster_manager
        self.clusters = cluster_manager.create_clusters(clusters_discr)

    def set_link_with_clusters(self, attribute):
        """Устанавливает связь между семантическими кластерами."""
        grouped_attribute = inseq_helpers.group_by(attribute, gmm_norm=True)
        return self._create_parsed_attribution(grouped_attribute)

    def get_cluster_importance_df(self, attribute):
        """Преобразует атрибуты в dataframe."""
        attribute_with_clusters = self.set_link_with_clusters(attribute)
        return inseq_helpers.attr_to_df(attribute_with_clusters)

    def display_attr(self, attribute):
        """Отображает атрибуты в виде тепловой карты."""
        attribute_with_clusters = self.set_link_with_clusters(attribute)
        return viz.attr_to_heatmap(attribute_with_clusters)

    def _create_parsed_attribution(self, grouped_attribute: AttrObj):
        """Создает разобранные атрибуции с сгенерированными метками."""
        tokens_generated_cl = [self.cluster_manager.find_cluster_name(word=x, clusters=self.clusters)
                               for x in grouped_attribute.tokens_generated]

        tokens_input_cl = [self.cluster_manager.find_cluster_name(word=x, clusters=self.clusters)
                           for x in grouped_attribute.tokens_input]

        grouped_attribute = copy.deepcopy(grouped_attribute)
        grouped_attribute.tokens_input = tokens_input_cl
        grouped_attribute.tokens_generated = tokens_generated_cl
        return grouped_attribute


def aggregate_cluster_df(cluster_df: DataFrame,
                         aggr_f: Union[('max', 'min', 'sum', 'mean', 'median', 'std', 'var', 'sem', 'skew')],
                         drop_condition: str = 'unnamed',
                         cl_names_col: str = 'Tokens',
                         ) -> DataFrame:
    cluster_df = cluster_df[~cluster_df[cl_names_col].str.contains(drop_condition)]
    cluster_df = cluster_df.loc[:, ~cluster_df.columns.str.contains(drop_condition)]
    aggregation_functions = {col: aggr_f for col in cluster_df.columns if col != cl_names_col}
    return cluster_df.groupby(cl_names_col).agg(aggregation_functions).reset_index()
