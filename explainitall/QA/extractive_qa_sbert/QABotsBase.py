import pickle
import re
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors



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


class KnnBot:
    """
    Поисковый бот на базе модели векторизации и метода ближайших соседей
    с установкой максимального радиуса для детекции аномалий.
    """

    def __init__(self, knn=None, sbert=None, mean=None, std=None, vect_transformer=None, dim=None, n_neighbors=3, eps=1e-200):
        self.knn = knn
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
        Преобразование текста в вектор и его нормализация.
        """
        vect_q = self.model.encode(q, convert_to_tensor=False)
        vect_q = self.vect_transformer.transform([vect_q])[0]
        return (vect_q - self.mean) / self.std
    
    def __get_answer_text(self, text_q, n_neighbors=1):
        """
        Получение ответов на основе векторизованного текста.
        
        :param text_q: Векторизованный запрос.
        :param n_neighbors: Количество ближайших соседей для поиска.
        :return: Список ответов.
        """
        vect = self.get_vect(text_q)
        
        # Получаем индексы ближайших соседей
        distances, indices = self.knn.kneighbors([vect], n_neighbors=n_neighbors)
        
        # Возвращаем список ответов на основе индексов ближайших соседей
        return list(set([self.answers[idx] for idx in indices[0]]))

    def get_answer(self, q, n_neighbors=1):
        """
        Получение ответа на входящий запрос.
        
        :param q: Вопрос.
        :param n_neighbors: Количество ближайших соседей, которые нужно вернуть.
        :return: Список ответов.
        """
        text_q = self.clean_string(q)
        return self.__get_answer_text(text_q, n_neighbors)

    def train(self, csv_path, embedder, knn_neighbors=5):
        """
        Метод для обучения бота на основе CSV с вопросами и ответами с расчетом среднего и стандартного отклонения.

        :param csv_path: Путь к CSV файлу с вопросами и ответами.
        :param embedder: Модель эмбеддинга для преобразования текста в векторы.
        :param knn_neighbors: Количество соседей для метода KNN.
        """
        # Шаг 1: Загрузка данных из CSV
        data = pd.read_csv(csv_path)
        if 'question' not in data.columns or 'answer' not in data.columns:
            raise ValueError("CSV файл должен содержать колонки 'question' и 'answer'")

        questions = data['question'].tolist()
        answers = data['answer'].tolist()

        # Шаг 2: Преобразование вопросов в векторы
        question_vectors = np.array([embedder.encode(q, convert_to_tensor=False) for q in questions])

        # Шаг 3: Вычисление среднего и стандартного отклонения по векторам вопросов
        self.mean = np.mean(question_vectors, axis=0)
        self.std = np.std(question_vectors, axis=0)

        # Чтобы избежать деления на ноль, добавляем небольшое значение eps
        eps = 1e-10
        self.std += eps

        # Шаг 4: Нормализация векторов вопросов
        normalized_vectors = (question_vectors - self.mean) / self.std

        # Шаг 5: Инициализация и обучение KNN
        self.knn = NearestNeighbors(n_neighbors=knn_neighbors, metric='cosine')
        self.knn.fit(normalized_vectors)

        # Сохранение ответов
        self.answers = answers

    def get_normalized_vector(self, question):
        """
        Получение нормализованного вектора вопроса на основе сохраненных среднего и std.
        
        :param question: Вопрос для преобразования.
        :return: Нормализованный вектор.
        """
        vect_q = self.model.encode(question, convert_to_tensor=False)
        return (vect_q - self.mean) / self.std
