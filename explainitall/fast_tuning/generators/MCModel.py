import pickle
import numpy as np


# преобразование 2х токенов в строку
def token_pair_2_str(tokens):
    list_x = list(tokens)
    return str(list_x)

# Кодирование конца
def encode_end(tokens_gpt, x_encode):
    str_x = token_pair_2_str(tokens_gpt[-2:])
    return x_encode[str_x] if str_x in x_encode else -1

# Markov
class MarkovModel():
    def __init__(self, len_vect, x_e=None, y_d = None, model = None, path = None, dep = 4):
        if path == None:
          self.x_encoder = x_e
          self.y_decoder = y_d
        else:
          with open(path + 'x_enc.dat', 'rb') as f:
            self.x_encoder = pickle.load(f)
          with open(path + 'y_dec.dat', 'rb') as f:
            self.y_decoder = pickle.load(f)
          with open(path + 'model.dat', 'rb') as f:
            model = pickle.load(f)

        self.dep = dep
        self.model_log = self.__get_log_model(model)
        self.len_vect = len_vect

    #----------------------- Генрерация bias --------------#
    def get_bias(self, token_1 = 1, token_2 = 1):
        '''Генерация bias'''
        b = [token_1, token_2]
        key = encode_end(b, self.x_encoder)
        bias = np.zeros((self.len_vect))

        if key != -1:
            tokens, logs = self.model_log[key]['tokens'], self.model_log[key]['logists']
            tokens = self.__get_tokens(tokens)
            bias  -= self.dep

            for i, token in enumerate(tokens):
                bias[token] = logs[i]

        return bias

    # Получение нормированных логарифмов
    def __get_logist(self, key, model):
        logist = np.log(model[key]['probs'])
        logist-=max(logist)
        return logist

    # Получение модели с логарифмами вероятностей
    def __get_log_model(self, model):
        m_l ={}
        for key in range(len(model)):
            m_l_semple = {}
            m_l_semple.update({'tokens': model[key]['tokens']})
            m_l_semple.update({'logists':self.__get_logist(key, model)})
            m_l.update({key:m_l_semple})
        return m_l


    def __get_tokens(self, tokens):
        true_tokens = []
        for t in tokens:
            true_tokens.append(self.y_decoder[t])
        return true_tokens


