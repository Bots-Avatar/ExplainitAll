import asyncio
import os
from typing import List

import gensim
import numpy as np
from fastapi import FastAPI
from inseq import load_model
from pydantic import BaseModel, constr
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from starlette.responses import RedirectResponse

from explainitall.QA.extractive_qa_sbert.QABotsBase import cos_dist
from explainitall.QA.extractive_qa_sbert.SVDBert import SVDBertModel
from explainitall.QA.interp_qa.KNNWithGenerative import FredStruct, PromptBot
from explainitall.gpt_like_interp import interp
from explainitall.gpt_like_interp.downloader import DownloadManager


class GetAnswerItem(BaseModel):
    question: constr(min_length=1)
    top_k: int


class LoadDatasetItem(BaseModel):
    questions: List[str]
    answers: List[str]


class ClusterItem(BaseModel):
    name: constr(min_length=1)
    centroid: List[str]
    top_k: int


class EvaluationItem(BaseModel):
    nlp_model_path: constr(min_length=1)
    nn_model_path: constr(min_length=1)
    clusters: List[ClusterItem]
    prompt: constr(min_length=1)
    generated_text: constr(min_length=1)


class ApiServerInit:
    def __init__(self, sbert_path='FractalGPT/SbertSVDDistil', device='cpu'):
        self.clusters_r = None
        self.bot = None
        self.prompt = None
        self.generated_text = None
        self.explainer = None
        self.clusters = None
        self.device = device
        self.app = FastAPI()
        self.sbert = SentenceTransformer(sbert_path)
        self.sbert[0].auto_model = SVDBertModel.from_pretrained(sbert_path)
        # self.fred = FredStruct('SiberiaSoft/SiberianFredT5-instructor')

        if os.getenv('TEST_MODE_ON_LOW_SPEC_PC') == 'True':
            self.fred = FredStruct('ai-forever/FRED-T5-large')
        else:
            self.fred = FredStruct('FractalGPT/FRED-T5-Interp')

    def load_dataset(self, questions, answers):
        self.__init_knn__(questions, answers)
        self.bot = PromptBot(self.knn, self.sbert, self.fred, answers, device=self.device)
        return True

    @staticmethod
    def df_to_dict(data_frame):
        df_copy = data_frame.copy(deep=True)

        def make_columns_unique(df):
            new_columns = {}
            for column in df.columns:
                if column in new_columns:
                    new_columns[column] += 1
                    new_name = f"{column}_{new_columns[column]}"
                else:
                    new_columns[column] = 0
                    new_name = column
                yield new_name

        df_copy.columns = list(make_columns_unique(df_copy))
        return df_copy.replace([np.nan, np.inf, -np.inf], ["nan", "inf", "-inf"]).to_dict(orient="split")

    def evaluate(self, nlp_model_path, nn_model_path, clusters, prompt, generated_text):
        self.clusters_r = [{
            "name": cluster.name,
            "centroid": cluster.centroid,
            "top_k": cluster.top_k
        } for cluster in clusters]
        self.prompt = prompt
        self.generated_text = generated_text
        self.__load_nlp_model__(nlp_model_path)
        self.__load_nn_model__(nn_model_path)
        self.explainer = interp.ExplainerGPT2(gpt_model=self.nn_model, nlp_model=self.nlp_model)
        expl_data = self.explainer.interpret(
            input_texts=self.prompt,
            generated_texts=self.generated_text,
            clusters_description=self.clusters_r,
            batch_size=50,
            steps=34,
            # max_new_tokens=19
        )
        return {"word_importance_map": self.df_to_dict(expl_data.word_imp_df),
                "word_importance_map_normalized": self.df_to_dict(expl_data.word_imp_norm_df),
                "cluster_importance_map": self.df_to_dict(expl_data.cluster_imp_df),
                "cluster_importance_map_normalized": self.df_to_dict(expl_data.cluster_imp_aggr_df)}

    def get_answer(self, q, top_k):
        return self.bot.get_answers(q, top_k=top_k)

    def __init_knn__(self, questions, answers):
        vects_questions = self.sbert.encode(questions)
        m = vects_questions.mean(axis=0)
        s = vects_questions.std(axis=0)
        knn_vects_questions = (vects_questions - m) / s

        self.knn = KNeighborsClassifier(metric=cos_dist)
        self.knn.fit(knn_vects_questions, answers)

    def __load_nlp_model__(self, url):
        self.nlp_model_url = url
        nlp_model_path = DownloadManager.load_zip(url)
        self.nlp_model = gensim.models.KeyedVectors.load_word2vec_format(nlp_model_path, binary=True)
        return True

    def __load_nn_model__(self, model_name_or_path):
        path = os.path.normpath(model_name_or_path)
        path_list = path.split(os.sep)
        self.nn_model_name = path_list[-1]
        self.nn_model = load_model(model=model_name_or_path, attribution_method="integrated_gradients")
        return True


api_server_init = ApiServerInit()
app = api_server_init.app


@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.post(
    "/load_dataset",
    summary="Load a dataset for the Q&A model",
    response_description="Indicates success of dataset loading",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example1": {
                            "summary": "Загрузка базового набора данных вопросов и ответов",
                            "value": {
                                "questions": ["Что такое коала?",
                                              "Опишите африканского слона"],

                                "answers": ["Это вид медведей, обитающих в Австралии.",
                                            "Это крупное млекопитающее с длинным хоботом."]
                            }
                        }
                    }
                }
            }
        }
    }
)
async def load_dataset(item: LoadDatasetItem):
    """Загружает набор данных, состоящий из вопросов и ответов о животных, в модель вопросов и ответов"""
    result = await asyncio.to_thread(api_server_init.load_dataset, item.questions, item.answers)
    return {"result": result}


@app.post(
    "/get_answer",
    summary="Получить ответы на вопросы",
    response_description="Полученный(е) ответ(ы) на указанные вопросы",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example1": {
                            "summary": "Получить ответ на вопрос о животном",
                            "value": {
                                "question": "Что за животное коала?",
                                "top_k": 1
                            }
                        }
                    }
                }
            }
        }
    }
)
async def get_answer(item: GetAnswerItem):
    """
    Получает ответы на указанный вопрос с использованием загруженной модели вопросов и ответов
    """
    result = await asyncio.to_thread(api_server_init.get_answer, item.question, item.top_k)
    return {"result": result}


@app.post(
    "/evaluate",
    summary="Оценить сгенерированный текст с использованием модели",
    response_description="Результаты оценки",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example1": {
                            "summary": "Оценить интерпретируемость текста",
                            "value": {
                                "nlp_model_path": 'http://vectors.nlpl.eu/repository/20/180.zip',
                                "nn_model_path": "ai-forever/rugpt3small_based_on_gpt2",
                                "clusters": [
                                    {'name': 'Животные', 'centroid': ['собака', 'кошка', 'заяц'], 'top_k': 140},
                                    {'name': 'Лекарства', 'centroid': ['уколы', 'таблетки', 'противовирусное'],
                                     'top_k': 160},
                                    {'name': 'Болезни', 'centroid': ['простуда', 'орви', 'орз', 'грипп'], 'top_k': 20},
                                    {'name': 'Симптомы', 'centroid': ['температура', 'насморк'], 'top_k': 20}
                                ],
                                "prompt": "я думаю что у моей кошки простуда, у нее температура, постоянный кашель: чем мне лечить мою кошку? ответ:",
                                "generated_text": "На сегодняшний день существует специальное противовирусное лечение для кошек, так же можно применять антибиотики"
                            }
                        }
                    }
                }
            }
        }
    }
)
async def evaluate(item: EvaluationItem):
    """Оценивает интерпретируемость сгенерированного текста относительно входного подсказки и предоставленных кластеров"""
    data = await asyncio.to_thread(api_server_init.evaluate, item.nlp_model_path, item.nn_model_path, item.clusters,
                                   item.prompt, item.generated_text)
    return data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
