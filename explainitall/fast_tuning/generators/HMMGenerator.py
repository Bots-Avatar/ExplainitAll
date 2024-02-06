import torch
import pickle
import numpy as np
import torch.nn.functional as F

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

# Class
class GPTGenerator():

  def __init__(self, model, tokenizer, hmm, device = 'cpu'):
    self.infer_device = device
    self.model_gpt = model
    self.model_gpt.to(device)
    self.tokenizer_gpt = tokenizer
    self.hmm = hmm

  # Штраф за повтор
  def __repeat_penalty(self, generated, logist, rp):
    hist_tokens = list(generated[0].cpu().numpy())
    set_tokens = set(hist_tokens)
    for token, log in enumerate(logist):
      if token in set_tokens:
        logist[token] = log-rp*(hist_tokens.count(token))


  #top p, top k фильтрация
  def _get_token(self, logist, top_k = 30, top_p = 0.9, del_simbols = None):

    filter_value = float('-inf')

    # Фильтруем нежелательные токены
    if del_simbols != None:
      for del_s in del_simbols:
        logist[del_s] = filter_value

    #top_k
    topk = torch.topk(logist, top_k)
    topk_v = topk[0]  # значения top_k
    topk_i = topk[1]  # индексы top_k

    #top_p
    probs =  F.softmax(topk_v, dim = -1)
    cumulative_probs = torch.cumsum(probs, dim = -1)
    probs[top_p < cumulative_probs] = 0

    if sum(probs) == 0:
      probs[0] = 1

    token_ind = torch.multinomial(probs, 1)
    token = topk_i[token_ind]
    return token

  # Генерация последовательности
  def _sample_sequence(self, length, context_tokens, temperature=1, top_k=30, hmm_w = 0.2, rp = 0.1, del_simbols = None):
      inp_len = len(context_tokens)
      context = torch.tensor(context_tokens, dtype=torch.long, device=self.infer_device)
      generated = context.unsqueeze(0)

      with torch.no_grad():
          decoded = ''
          for _ in range(length):
              inputs = {'input_ids': generated[:, -1023:]} # Входы
              outputs = self.model_gpt(**inputs) # Прямой проход gpt

              g_with_start = [1,1]+list(generated[0][inp_len:].cpu().numpy())# Затравка для hmm
              bias_hmm = self.hmm.get_bias(g_with_start[-2:]) # bias на базе hmm
              bias_hmm = torch.tensor(bias_hmm).to(self.infer_device) # bias на базе hmm в виде тензора
              next_token_logits = ((1-hmm_w)*outputs[0][0, -1, :]+hmm_w*bias_hmm) / temperature
              self.__repeat_penalty(generated, next_token_logits, rp)
              next_tokens = self._get_token(next_token_logits, top_k = top_k, del_simbols = del_simbols).to(self.infer_device) # Генерация из распределения

              if next_tokens == 0:
                break

              generated = torch.cat((generated, next_tokens.unsqueeze(-1)), dim=1)

          out = generated[0, len(context_tokens):].tolist()
          new_decoded = self.tokenizer_gpt.decode(out)
          if len(new_decoded) > len(decoded):
              decoded = new_decoded
      return decoded


  # Генерация текста из текста
  def _generate(self, raw_text, length=250, temperature=1., top_k=30, rp = 0.03, hmm_w =0.2, del_simbols = None):
      context_tokens = self.tokenizer_gpt.encode(raw_text)
      out = self._sample_sequence(
          length, context_tokens,
          rp = rp,
          hmm_w=hmm_w,
          temperature=temperature,
          top_k=top_k,
          del_simbols = del_simbols,
      )
      return out

  # Генерация нескольких последовательностей из начального текста
  def Generate(self, text, max_len = 200, num_seq = 2, temperature = 1.0, topk = 30, rp = 0.03, hmm_w =0.2, del_simbols = None):
    ret = []
    for i in range(num_seq):
      ret.append(
          self._generate(text, max_len, temperature=temperature, top_k=topk, rp=rp, hmm_w = hmm_w, del_simbols=del_simbols)
          )
    return ret

##################################################################################################################################################################