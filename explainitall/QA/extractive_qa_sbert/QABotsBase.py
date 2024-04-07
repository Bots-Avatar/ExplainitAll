import pickle
import re

import numpy as np


class RetriBotStruct:
    """Ретривел бот"""

    def __init__(self, path=None, knn=None, embedder=None, answers=None):

        if path is None:
            if not (knn is None or embedder is None or answers is None):
                self.knn = knn
                self.embedder = embedder
                self.answers = answers

            else:
                print('Ошибка! Укажите путь или метод ближ. соседа, ответы и эмбеддер')

        else:
            self.load(path)

    def to(self, device='cpu'):
        self.embedder.to(device)

    def get_knn(self):
        """Возвращает knn"""
        return self.knn

    def get_embedder(self):
        """Возвращает эбеддер"""
        return self.embedder

    def get_ans_list(self):
        """Возвращает список ответов"""
        return self.answers

    def save(self, path):
        """Сохранить"""
        self.to()
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        """Загрузка"""
        with open(path, 'rb') as f:
            retri_bot = pickle.load(f)
        self.knn = retri_bot.knn
        self.embedder = retri_bot.embedder
        self.answers = retri_bot.answers


class QABotStruct:
    """Вопрос-ответный бот"""

    def __init__(self, path=None, retri_bot=None, qa=None):

        if path == None:
            if not (retri_bot == None or qa == None):
                self.retri_bot = retri_bot
                self.qa = qa

            else:
                print('Ошибка! Укажите путь или ретривел бот и QA систему')

        else:
            self.load(path)

    def get_retri_bot(self):
        """Возвращает ретривел бот"""
        return self.retri_bot

    def get_qa(self):
        """Возвращает QA систему"""
        return self.qa

    def save(self, path):
        """Сохранить"""
        self.qa.model.to('cpu')
        self.retri_bot.to('cpu')
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        """Загрузка"""
        with open(path, 'rb') as f:
            qa_bot = pickle.load(f)

        self.qa = qa_bot.qa
        self.retri_bot = qa_bot.retri_bot


class RetriBot():
    """Ретривел бот"""

    def __init__(self, bot, max_words=50, device='cpu'):

        if 'str' in str(type(bot)):
            rBot = RetriBotStruct(bot)
        else:
            rBot = bot

        self.main_knn = rBot.get_knn()
        self.sModel = rBot.get_embedder().to(device)
        self.max_words = max_words
        self.texts = rBot.get_ans_list()

    def _get_vect(self, q):
        return self.sModel.encode(q, convert_to_tensor=False)

    @staticmethod
    def cut(text, max_len=15):
        words = text.split(' ')[:max_len]
        ret_text = ''
        for word in words:
            ret_text += word + ' '

        return ret_text

    def get_answers(self, q, top_k=7):
        vect_q = self._get_vect(q)
        ans = self.main_knn.kneighbors([vect_q], top_k)
        support = []

        for i in range(ans[0].shape[1]):
            support.append(self.texts[ans[1][0][i]])

        support = list(set(support))

        ret_line = ''

        for doc in support:
            ret_line += RetriBot.cut(doc, self.max_words) + '. '

        return ret_line


class QABot:
    """QA"""

    def __init__(self, bot, max_words=50, device='cpu'):

        if 'str' in str(type(bot)):
            qBot = QABotStruct(bot)
        else:
            qBot = bot

        self.retr = RetriBot(qBot.get_retri_bot(), max_words=max_words, device=device)
        self.qa = qBot.get_qa()
        self.qa.model.to(device)

    def qa_get_answer(self, text, q, top_k=3):
        ans = self.qa(context=text, question=q, top_k=top_k)
        answers = []

        if top_k == 1:
            return [{'answer': ans['answer'], 'score': ans['score']}]

        else:
            for a in ans:
                answers.append({'answer': a['answer'], 'score': a['score']})

        return answers

    def search(self, text, q, confidence=0.2):
        answer = self.qa_get_answer(text, q, 1)[0]
        if answer['score'] >= confidence:
            return answer['answer']
        else:
            return '<NoAnswer>'

    def get_prompt(self, q, confidence=0.3, top_k_search=7):
        text = self.retr.get_answers(q, top_k_search)
        return self.search(text, q, confidence)


class SimpleTransformer():
    """ Класс-заглушка """

    def __init__(self):
        pass

    def transform(self, vects):
        return vects


def cos(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def cos_dist(x, y):
    return -cos(x, y)


class KnnBot():
    """
    Поисковый бот на базе модели векторизации и метода ближайших соседей
    с установкой максимального радиуса для детекции аномалий.
    """

    def __init__(self, knn, sbert, mean=None, std=None, vect_transformer=None, dim=None, n_neighbors=3, eps=1e-200):
        self.knn = knn
        self.knn.n_neighbors = n_neighbors
        self.model = sbert
        self.mean = np.zeros((dim,)) if mean is None else mean
        self.std = np.ones((dim,)) if std is None else std + eps
        self.vect_transformer = SimpleTransformer() if vect_transformer is None else vect_transformer

        self.r_char = re.compile('[^A-zА-яЁё0-9": ]')
        self.r_spaces = re.compile(r"\s+")

    def clean_string(self, text):
        """
        Очистка и нормализация строки.
        """
        seq = self.r_char.sub(' ', text.replace('\n', ' '))
        seq = self.r_spaces.sub(' ', seq).strip()
        return seq.lower()

    def get_vect(self, q):
        """
        Преобразование текста в вектор.
        """
        vect_q = self.model.encode(q, convert_to_tensor=False)
        vect_q = self.vect_transformer.transform([vect_q])[0]
        return (vect_q - self.mean) / self.std

    def __get_answer_text(self, text_q):
        """
        Получение ответа на основе векторизованного текста.
        """
        vect = self.get_vect(text_q)
        return self.knn.predict([vect])[0]

    def get_answer(self, q):
        """
        Получение ответа на входящий запрос.
        """
        text_q = self.clean_string(q)
        return self.__get_answer_text(text_q)
