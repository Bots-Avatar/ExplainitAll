import torch
import torch.nn.functional as F


class GPTGenerator:

  def __init__(self, model, tokenizer, expert, device = None):
    self.infer_device = device
    self.model_gpt = model
    self.model_gpt.to(device)
    self.tokenizer_gpt = tokenizer
    self.expert = expert

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
  def _sample_sequence(self, length, context_tokens, temperature=1, top_k=30, expert_w = 0.2, rp = 0.1, del_simbols = None):
      inp_len = len(context_tokens)
      context = torch.tensor(context_tokens, dtype=torch.long, device=self.infer_device)
      generated = context.unsqueeze(0)

      with torch.no_grad():
          decoded = ''
          for _ in range(length):
              inputs = {'input_ids': generated[:, -1023:]} # Входы
              outputs = self.model_gpt(**inputs) # Прямой проход gpt

              g_with_start = list(generated[0].cpu().numpy())# Затравка для эксперта
              bias_expert = self.expert.get_bias(g_with_start) # bias на базе эксперта или их смеси
              bias_expert = torch.tensor(bias_expert).to(self.infer_device) # bias в виде тензора
              next_token_logits = ((1-expert_w)*outputs[0][0, -1, :]+expert_w*bias_expert) / temperature
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
  def _generate(self, raw_text, length=250, temperature=1., top_k=30, rp = 0.03, expert_w =0.2, del_simbols = None):
      context_tokens = self.tokenizer_gpt.encode(raw_text)
      out = self._sample_sequence(
          length, context_tokens,
          rp = rp,
          expert_w=expert_w,
          temperature=temperature,
          top_k=top_k,
          del_simbols = del_simbols,
      )
      return out

  # Генерация нескольких последовательностей из начального текста
  def Generate(self, text, max_len = 200, num_seq = 2, temperature = 1.0, topk = 30, rp = 0.03, expert_w =0.2, del_simbols = None):
    ret = []
    for i in range(num_seq):
      ret.append(
          self._generate(text, max_len, temperature=temperature, top_k=topk, rp=rp, expert_w = expert_w, del_simbols=del_simbols)
          )
    return ret

##################################################################################################################################################################