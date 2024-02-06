import os
import re
import math
import json
import copy
import pandas as pd
import itertools
import gradio as gr

import io
from PIL import Image

import tempfile

import gensim
from inseq import load_model

from explainitall.gpt_like_interp import viz
from explainitall.gpt_like_interp import dl
from explainitall.gpt_like_interp import interp

from explainitall.QA.interp_qa.KNNWithGenerative import FredStruct, PromptBot
from explainitall.QA.extractive_qa_sbert.SVDBert import SVDBertModel
from explainitall.QA.extractive_qa_sbert.QABotsBase import cos_dist
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
import gensim
from inseq import load_model
from explainitall.gpt_like_interp import viz
from explainitall.gpt_like_interp import dl
from explainitall.gpt_like_interp import interp

import matplotlib.pyplot as plt
import seaborn as sns

from .supporting_functions import df_to_heatmap_plot, make_dataframe_from_clusters, make_clusters_from_dataframe


def clean_string(text):
    """
    Очистка строки
    """
    seq = text.replace('\n', ' ')
    r_char = re.compile('[^A-zА-яЁё0-9": ]')
    r_spaces = re.compile(r"\s+")
    seq = r_char.sub(' ', seq)
    seq = r_spaces.sub(' ', seq).strip()
    return seq.lower()


def value_interp(v):
    if str(v) == 'nan':
        return 'нулевой'
    if v < 0.1:
        return 'незначительной'
    if v < 0.3:
        return 'очень малой'
    if v < 0.45:
        return 'малой'
    if v < 0.65:
        return 'средней'
    if v < 0.85:
        return 'выше средней'
    else:
        return 'очень большой'


def interp_cl(df):
    ret = []
    for index, row in df.iterrows():
        for num_col, col in enumerate(df.columns):
            if num_col != 0:
                value = row[col]

                description = f'Кластер "{row[df.columns[0]]}" влияет на генерацию кластера "{col}" с {value_interp(value)} силой.'
                ret += [description]

    return ret


class DemoInterface:
    def __init__(self):
        path_sbert = 'FractalGPT/SbertSVDDistil'
        self.sbert_ = SentenceTransformer(path_sbert)
        self.sbert_[0].auto_model = SVDBertModel.from_pretrained(path_sbert)

        self.fred_ = FredStruct()

        self.demo_ = gr.Blocks()

        with self.demo_:
            with gr.Tabs():
                with gr.TabItem("Texts"):
                    context_text = gr.Text(label='Context')
                    output_text = gr.Text(label='Generated text')
                    with gr.Row():
                        texts_load_button = gr.Button("Set texts")
                        texts_set_checkbox = gr.Checkbox(label='Texts are set', interactive=False)
                with gr.TabItem("Clusters"):
                    with gr.Tabs():
                        with gr.TabItem("Load clusters from file"):
                            with gr.Column():
                                clusters_file = gr.File(label='Clusters\' file')
                                with gr.Row():
                                    clusters_load_from_file_button = gr.Button("Load clusters")
                        with gr.TabItem("Set clusters manually"):
                            with gr.Column():
                                set_clusters_table = gr.Dataframe(label='Clusters\' table',
                                                                  headers=['name', 'centroid', 'top_k'],
                                                                  # max_rows=None,
                                                                  height=None,
                                                                  # overflow_row_behaviour='paginate',
                                                                  wrap=False,
                                                                  interactive=True)
                                with gr.Row():
                                    set_clusters_from_dataframe_button = gr.Button("Set clusters")

                    clusters_set_checkbox = gr.Checkbox(label='Clusters are set', interactive=False)

                    cluster_table = gr.Dataframe(label='Clusters',
                                                 headers=['name', 'centroid', 'top_k'],
                                                 # max_rows=None,
                                                 height=None,
                                                 # overflow_row_behaviour='paginate',
                                                 wrap=False,
                                                 interactive=False)

                    clusters_file_path_text = gr.Text(label='Save clusters to file')
                    with gr.Row():
                        clusters_save_button = gr.Button("Save clusters")
                        clusters_save_checkbox = gr.Checkbox(label='Clusters are saved', interactive=False)

                with gr.TabItem("NLP model"):
                    nlp_model_url = gr.Text(label='Model url')
                    with gr.Row():
                        load_nlp_model_button = gr.Button("Load model")
                        nlp_model_set_checkbox = gr.Checkbox(label='NLP model loaded', interactive=False)
                with gr.TabItem("LLM interpretation model"):
                    nn_model_name_or_path = gr.Text(label='Model name or path')
                    with gr.Row():
                        load_nn_model_button = gr.Button("Load model")
                        nn_model_set_checkbox = gr.Checkbox(label='NN model loaded', interactive=False)
                with gr.TabItem("Results"):
                    with gr.Row():
                        with gr.Column():
                            result_texts_set_checkbox = gr.Checkbox(label='Texts are set', interactive=False)
                            result_clusters_set_checkbox = gr.Checkbox(label='Clusters are set', interactive=False)
                            result_nlp_model_set_checkbox = gr.Checkbox(label='NLP model loaded', interactive=False)
                            result_nn_model_set_checkbox = gr.Checkbox(label='NN model loaded', interactive=False)
                            launch_button = gr.Button("Launch")
                        with gr.Column():
                            with gr.Tabs():
                                with gr.TabItem("Word importance"):
                                    # word_importance_image = gr.Image().style(height=600)
                                    word_importance_image = gr.Image(height=600)
                                with gr.TabItem("Word importance normalized"):
                                    # word_importance_norm_image = gr.Image().style(height=600)
                                    word_importance_norm_image = gr.Image(height=600)
                                with gr.TabItem("Cluster importance"):
                                    # cluster_importance_image = gr.Image().style(height=600)
                                    cluster_importance_image = gr.Image(height=600)
                                with gr.TabItem("Cluster importance grouped"):
                                    # cluster_importance_norm_image = gr.Image().style(height=600)
                                    cluster_importance_norm_image = gr.Image(height=600)
                with gr.TabItem("Chatbot"):
                    chat = gr.Chatbot(label="Chatbot", layout="bubble")
                    msg = gr.Text(label="Message", interactive=True)
                    send_message = gr.Button("Send", interactive=True)
                    clear = gr.ClearButton([msg, chat])
                    # chat.change(fn=chatbot_function, inputs=msg, outputs=chat)

            texts_load_button.click(self.load_context_and_generated_text_,
                                    inputs=[context_text, output_text],
                                    outputs=[texts_set_checkbox, result_texts_set_checkbox])

            clusters_load_from_file_button.click(self.load_clusters_from_file_,
                                                 inputs=[clusters_file],
                                                 outputs=[clusters_set_checkbox, result_clusters_set_checkbox,
                                                          cluster_table, set_clusters_table])
            set_clusters_from_dataframe_button.click(self.set_clusters_from_dataframe_,
                                                     inputs=[set_clusters_table],
                                                     outputs=[clusters_set_checkbox, result_clusters_set_checkbox,
                                                              cluster_table])
            clusters_save_button.click(self.save_new_clusters_to_file_,
                                       inputs=[cluster_table, clusters_file_path_text],
                                       outputs=[clusters_save_checkbox])

            load_nn_model_button.click(self.load_nn_model_,
                                       inputs=[nn_model_name_or_path],
                                       outputs=[nn_model_set_checkbox, result_nn_model_set_checkbox])

            load_nlp_model_button.click(self.load_nlp_model_,
                                        inputs=[nlp_model_url],
                                        outputs=[nlp_model_set_checkbox, result_nlp_model_set_checkbox])

            launch_button.click(self.show_results, inputs=[], outputs=[word_importance_image,
                                                                       word_importance_norm_image,
                                                                       cluster_importance_image,
                                                                       cluster_importance_norm_image])

            send_message.click(self.respond_, inputs=[msg, chat], outputs=[msg, chat])

    def __del__(self):
        pass

    # FUNCTIONALITY:

    def launch(self):
        self.demo_.launch()

    def show_results(self):
        self.explainer_ = interp.ExplainerGPT2(gpt_model=self.nn_model_, nlp_model=self.nlp_model_)
        expl_data = self.explainer_.interpret(input_texts=self.context_,
                                              generated_texts=self.generated_text_,
                                              clusters_description=self.clusters_,
                                              batch_size=50,
                                              steps=34,
                                              # max_new_tokens=19
                                              )

        # Результат интерпретации
        imp_df_cl = expl_data.cluster_imp_aggr_df
        cl_desc = interp_cl(imp_df_cl)

        clean = [clean_string(cl_data) for cl_data in cl_desc]
        vects_x = self.sbert_.encode(clean)
        m = vects_x.mean(axis=0)
        s = vects_x.std(axis=0)
        knn_vects_x = (vects_x - m) / s
        knn = KNeighborsClassifier(metric=cos_dist)
        knn.fit(knn_vects_x, cl_desc)

        self.interp_bot_ = PromptBot(knn, self.sbert_, self.fred_, cl_desc, device='cpu')

        word_importance_plt = df_to_heatmap_plot(expl_data.word_imp_df, title="Карта важности слов")
        word_importance_norm_plt = df_to_heatmap_plot(expl_data.word_imp_norm_df,
                                                      title="Карта важности слов, нормированная")

        cluster_importance_plt = df_to_heatmap_plot(expl_data.cluster_imp_df, title="Карта важности кластеров")
        cluster_importance_norm_plt = df_to_heatmap_plot(expl_data.cluster_imp_aggr_df,
                                                         title="Карта важности кластеров, группированная")

        # word_importance_image.style(width=256, height=256)
        # word_importance_norm_image.style(width=256, height=256)

        return word_importance_plt, word_importance_norm_plt, cluster_importance_plt, cluster_importance_norm_plt

    # PRIVATE FUNCTIONS:

    def respond_(self, message, chat_history):
        ans = self.interp_bot_.get_answers(message, top_k=3)
        bot_reply = ans.split('.')[0]
        chat_history.append((message, bot_reply))

        return "", chat_history

    def load_context_and_generated_text_(self, context, generated_text):
        self.context = context
        self.generated_text_ = generated_text

        return True, True

    def load_clusters_from_file_(self, jsonfile: tempfile._TemporaryFileWrapper):
        with open(jsonfile.name, 'r') as fp:
            self.clusters_ = json.load(fp)
        df = make_dataframe_from_clusters(self.clusters_)

        return True, True, df, df

    def set_clusters_from_dataframe_(self, df):
        self.clusters_ = make_clusters_from_dataframe(df)

        return True, True, df

    def save_new_clusters_to_file_(self, df, filename):
        clusters = make_clusters_from_dataframe(df)
        with open(filename, 'w') as fp:
            json.dump(clusters, fp)

        return True

    def load_nlp_model_(self, url):
        self.npl_model_url_ = url
        nlp_model_path = dl.DownloadManager.load_zip(url)
        self.nlp_model_ = gensim.models.KeyedVectors.load_word2vec_format(nlp_model_path, binary=True)

        return True, True

    def load_nn_model_(self, model_name_or_path):
        path = os.path.normpath(model_name_or_path)
        path_list = path.split(os.sep)
        self.nn_model_name_ = path_list[-1]

        self.nn_model_ = load_model(model=model_name_or_path,
                                    attribution_method="integrated_gradients")

        return True, True

    # FIELDS:

    context_ = None
    generated_text_ = None

    clusters_ = None

    npl_model_url_ = None
    nlp_model_ = None

    nn_model_name_ = None
    nn_model_ = None

    explainer_ = None

    sbert_ = None

    fred_ = None

    interp_bot_ = None

    demo_: gr.blocks.Blocks = None
