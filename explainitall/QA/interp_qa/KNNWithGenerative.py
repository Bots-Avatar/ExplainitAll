import re

import nltk
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from explainitall.QA.extractive_qa_sbert.QABotsBase import SimpleTransformer

nltk.download('punkt')


class FredStruct:

    def __init__(self, path='Ponimash/FredInterpreter', device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.fred_model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.fred_model = self.fred_model.to(device)
        self.fred_model.eval()

    def get_model(self):
        return self.fred_model

    def get_tokenizer(self):
        return self.tokenizer


class PromptBot:
    """
  Поисковый бот на базе модели векторизации и метода ближ. соседа
  с установкой максимального радиуса для детекции аномалий
  """

    def __init__(self, knn, sbert, fred, texts, max_words=50, mean=None, std=None, vect_transformer=None, dim=None,
                 n_neighbors=3, eps=1e-200, device='cuda'):
        self.max_words = max_words
        self.knn = knn
        self.knn.n_neighbors = n_neighbors
        self.texts = texts
        self.sbert = sbert
        self.mean = np.zeros((dim)) if str(type(mean)) != "<class 'numpy.ndarray'>" else mean
        self.std = np.ones((dim)) if str(type(std)) != "<class 'numpy.ndarray'>" else std + eps
        self.vect_transformer = SimpleTransformer() if vect_transformer == None else vect_transformer
        self.fred = fred
        self.device = device

    def __clean_string(self, text):
        """
        Очистка строки
        """
        seq = text.replace('\n', ' ')
        r_char = re.compile('[^A-zА-яЁё0-9": ]')
        r_spaces = re.compile(r"\s+")
        seq = r_char.sub(' ', seq)
        seq = r_spaces.sub(' ', seq).strip()
        return seq.lower()

        return data_inp, data_outp

    def __qa__(self, doc, q):
        doc = doc.replace('\n', ' ')
        q = q.replace('\n', ' ')
        q_pr = f"<SC6>Опираясь только на информацию: {doc}.\n Подробно и полно ответь на вопрос указав все детали и числовые значения: \"{q}\" "
        data_inp = self.fred.get_tokenizer()(q_pr, return_tensors="pt").to(self.device)
        return data_inp

    def __generate__(self, doc, q):
        t = self.__qa__(doc, q)
        output_ids = self.fred.get_model().generate(
            **t, do_sample=True, temperature=0.2, max_new_tokens=256, top_p=0.95, top_k=15, repetition_penalty=1.03,
            no_repeat_ngram_size=3
        )[0]
        out = self.fred.get_tokenizer().decode(output_ids.tolist(), skip_special_tokens=True)
        return out.replace("<extra_id_0>", "")

    def get_vect(self, q):
        """вектор из текста"""
        vect_q = self.sbert.encode(q, convert_to_tensor=False)
        vect_q = self.vect_transformer.transform([vect_q])[0]
        return (vect_q - self.mean) / self.std

    @staticmethod
    def cut(text, max_len=15):
        words = text.split(' ')[:max_len]
        ret_text = ''
        for word in words:
            ret_text += word + ' '

        return ret_text

    def get_answers(self, q, top_k=7):
        """ответ по преобразованному тексту"""
        vect_q = self.get_vect(q)
        answer_ids = self.knn.kneighbors([vect_q], top_k)
        doc = ''
        for i in range(answer_ids[0].shape[1]):
            doc += self.texts[answer_ids[1][0][i]] + '\n'
        return self.__generate__(doc, q)
