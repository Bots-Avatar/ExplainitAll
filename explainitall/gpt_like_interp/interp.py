from typing import List, Dict

import gensim
import pandas as pd
from inseq import AttributionModel

from explainitall import clusters
from explainitall.clusters import ClusterManager, aggregate_cluster_df
from explainitall.gpt_like_interp import viz
from . import inseq_helpers
from .inseq_helpers import AttrObj


class ExplainerGPT2Output:
    def __init__(self, attributions: AttrObj,
                 attributions_grouped: AttrObj,
                 attributions_grouped_norm: AttrObj,

                 cluster_imp_df: pd.DataFrame,
                 cluster_imp_aggr_df: pd.DataFrame,
                 word_imp_df: pd.DataFrame,
                 word_imp_norm_df: pd.DataFrame):
        self.attributions = attributions
        self.attributions_grouped = attributions_grouped
        self.attributions_grouped_norm = attributions_grouped_norm

        self.cluster_imp_df = cluster_imp_df
        self.cluster_imp_aggr_df = cluster_imp_aggr_df
        self.word_imp_df = word_imp_df
        self.word_imp_norm_df = word_imp_norm_df

    def show_word_imp_heatmap(self):
        viz.df_to_heatmap(self.word_imp_df, title="Карта важности слов")

    def show_word_imp_norm_heatmap(self):
        viz.df_to_heatmap(self.word_imp_norm_df, title="Карта важности слов, нормированная")

    def show_cluster_imp_heatmap(self):
        viz.df_to_heatmap(self.cluster_imp_df, title="Карта важности слов")

    def show_cluster_imp_aggr_heatmap(self):
        viz.df_to_heatmap(self.cluster_imp_aggr_df, title="Карта важности слов группированная")


class ExplainerGPT2:
    def __init__(self, gpt_model: AttributionModel,
                 nlp_model: gensim.models.keyedvectors.KeyedVectors):
        self.gpt_model = gpt_model
        self.nlp_model = nlp_model
        self._cluster_manager = ClusterManager(embeddings=self.nlp_model)
        self.attributions = None

    def interpret(self,
                  input_texts: str,
                  generated_texts: str,
                  clusters_description: List[Dict[str, object]],
                  batch_size: int = 50,
                  steps: int = 34,
                  max_new_tokens: int = None,
                  aggr_f='mean') -> ExplainerGPT2Output:
        self._attribute(input_texts, generated_texts, max_new_tokens, steps, batch_size)
        return self._run_pipeline(clusters_description, aggr_f)

    @staticmethod
    def calc_max_tokes(input_texts, generated_texts):
        from gensim.utils import tokenize
        tokens = list(tokenize(input_texts + generated_texts, lowercase=True))
        num_tokens = len(tokens)
        return num_tokens + 10  # buffer

    def _attribute(self, input_texts: str,
                   generated_texts: str,
                   max_new_tokens: int,
                   steps: int, batch_size: int):

        generation_args = None
        if not generated_texts:
            if max_new_tokens is None:
                max_new_tokens = self.calc_max_tokes(input_texts, generated_texts)
                generation_args = {"max_new_tokens": max_new_tokens}

        out = self.gpt_model.attribute(
            input_texts=input_texts, generated_texts=input_texts + generated_texts,
            n_steps=steps, generation_args=generation_args,
            show_progress=True, pretty_progress=False, internal_batch_size=batch_size
        )
        self.attributions = inseq_helpers.get_first_attribute(out)

    def _run_pipeline(self, cluster_desc, aggr_f):
        group_attr = inseq_helpers.group_by(self.attributions)
        norm_attr = inseq_helpers.group_by(self.attributions, gmm_norm=True)

        word_imp_df = inseq_helpers.attr_to_df(group_attr)
        word_imp_norm_df = inseq_helpers.attr_to_df(norm_attr)

        cluster_interpreter = clusters.ClusterInterpreter(clusters_discr=cluster_desc,
                                                          cluster_manager=self._cluster_manager)

        cluster_imp_df = cluster_interpreter.get_cluster_importance_df(self.attributions)
        try:
            cluster_imp_aggr_df = aggregate_cluster_df(cluster_imp_df, aggr_f=aggr_f)
        except Exception as e:
            print(f"Неверно заданы выходные кластеры: {e}")
            cluster_imp_aggr_df = pd.DataFrame()

        return ExplainerGPT2Output(
            attributions=self.attributions, attributions_grouped=group_attr, attributions_grouped_norm=norm_attr,
            cluster_imp_df=cluster_imp_df, cluster_imp_aggr_df=cluster_imp_aggr_df,
            word_imp_df=word_imp_df, word_imp_norm_df=word_imp_norm_df)
