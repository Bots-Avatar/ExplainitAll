from explainitall.QA.interp_qa.KNNWithGenerative import FredStruct, PromptBot
from explainitall.QA.extractive_qa_sbert.SVDBert import SVDBertModel
from explainitall.QA.extractive_qa_sbert.QABotsBase import cos_dist
from explainitall.gpt_like_interp import dl
from explainitall.gpt_like_interp import interp
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from pydantic import BaseModel, constr
from inseq import load_model
from fastapi import FastAPI
from typing import List
import asyncio
import gensim
import os


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


class TestItem(BaseModel):
    nlp_model_path: constr(min_length=1)
    nn_model_path: constr(min_length=1)
    clusters: List[ClusterItem]
    prompt: constr(min_length=1)
    generated_text: constr(min_length=1)


class ApiServerInit:
    def __init__(self, sbert_path='FractalGPT/SbertSVDDistil', device='cpu'):
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
        self.fred = FredStruct()

    def load_dataset(self, questions, answers):
        self.__init_knn__(questions, answers)
        self.bot = PromptBot(self.knn, self.sbert, self.fred, answers, device=self.device)
        return True

    def test(self, nlp_model_path, nn_model_path, clusters, prompt, generated_text):
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

        return {"word_importance_map": expl_data.word_imp_df.to_json(orient="split"),
                "word_importance_map_normalized": expl_data.word_imp_norm_df.to_json(orient="split"),
                "cluster_importance_map": expl_data.cluster_imp_df.to_json(orient="split"),
                "cluster_importance_map_normalized": expl_data.cluster_imp_aggr_df.to_json(orient="split")}

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
        nlp_model_path = dl.DownloadManager.load_zip(url)
        self.nlp_model = gensim.models.KeyedVectors.load_word2vec_format(nlp_model_path, binary=True)
        return True

    def __load_nn_model__(self, model_name_or_path):
        path = os.path.normpath(model_name_or_path)
        path_list = path.split(os.sep)
        self.nn_model_name = path_list[-1]
        self.nn_model = load_model(model=model_name_or_path,
                                   attribution_method="integrated_gradients")
        return True


api_server_init = ApiServerInit()
app = api_server_init.app


@app.post("/load_dataset")
async def load_dataset(item: LoadDatasetItem):
    result = await asyncio.to_thread(api_server_init.load_dataset, item.questions, item.answers)
    return {"result": result}


4@app.post("/get_answer")
async def get_answer(item: GetAnswerItem):
    result = await asyncio.to_thread(api_server_init.get_answer, item.question, item.top_k)
    return {"result": result}


@app.post("/test")
async def test(item: TestItem):
    result = await asyncio.to_thread(api_server_init.test, item.nlp_model_path, item.nn_model_path,
                                     item.clusters, item.prompt, item.generated_text)
    return {"result": result}
