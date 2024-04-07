import numpy as np
import torch
from transformers import GPT2Model, GPT2Config


class GPTEmbedder:
    """Эмбеддер для gpt"""

    def __init__(self, tokenizer_gpt, gpt, device=None):

        if device is None:
            device = 'cpu'
            if torch.cuda.is_available():
                device = 'cuda:0'

        self.model = gpt
        self.model.to(device)
        self.tokenizer = tokenizer_gpt
        self.device = device

    #
    def get_emb_from_gpt(self, inp_str, n_layer_index=-1, token=-1, is_attention=False):
        """
        Данные(эмбеддинги) со скрытых слоев
        Вернуть скрытое состояние или данные внимания
        """
        att_hidden = 0
        if is_attention:
            att_hidden = 1

        inp_tokens = self.tokenizer.encode(inp_str)
        context = torch.tensor(inp_tokens, dtype=torch.long, device=self.device)
        generated = context.unsqueeze(0)
        inputs = {'input_ids': generated}

        with torch.no_grad():
            outputs = self.model(**inputs)

        if n_layer_index == 'all':
            return outputs.last_hidden_state[0, -1, :].reshape(self.model.config.n_embd).to('cpu').detach().numpy()
        else:
            return_obj = outputs[1][n_layer_index][att_hidden][0, :, token, :].to('cpu')  # 0 - т.к. батч ожидается 1

        shape = return_obj.shape
        return return_obj.reshape((shape[0] * shape[1])).detach().numpy()

    def get_embs_from_gpt(self, inp_str, n_layer_index=-1, head_index=0, is_attention=False):
        """
        Данные(эмбеддинги) со скрытых слоев(По всем токенам)
        Вернуть скрытое состояние или данные внимания
        """

        att_hidden = 0
        if is_attention:
            att_hidden = 1

        inp_tokens = self.tokenizer.encode(inp_str)
        context = torch.tensor(inp_tokens, dtype=torch.long, device=self.device)
        generated = context.unsqueeze(0)
        inputs = {'input_ids': generated}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Пройти всю сеть включая слой нормализации
        if n_layer_index == 'all':
            return outputs.last_hidden_state[0, :, :].reshape((len(inp_tokens), self.model.config.n_embd)).to(
                'cpu').detach().numpy()

        # Пройти заданное число gpt блоков
        else:
            # Вернуть все головы внимания или только 1
            if head_index == 'all':
                return_obj = outputs[1][n_layer_index][att_hidden][0, :, :, :].to('cpu').detach().numpy()
                out_len = return_obj.shape[0] * return_obj.shape[2]
                return_obj = np.transpose(return_obj, (1, 0, 2)).reshape(return_obj.shape[1], out_len)
            else:
                return_obj = outputs[1][n_layer_index][att_hidden][0, head_index, :, :].to(
                    'cpu').detach().numpy()  # 0 - т.к. батч ожидается 1

        return return_obj

    def _get_k_layer(self, num_layers=3, name='gpt-embeder'):
        """Создание модели на базе первых слоев gpt2 донора"""
        base_model = self.model.base_model  # модель-донор
        config_base = base_model.config  # Конфиг модели-донора
        config = GPT2Config.from_dict(config_base.to_dict())  # Копирование конфига

        if num_layers < 0:
            num_layers = config.n_layer + num_layers + 1

        config.name_or_path = name  # Имя сети
        config.n_layer = num_layers  # Установка нужного числа слоев

        gpt_emb = GPT2Model(config)  # Создание модели
        gpt_emb.wte.weight = self.model.transformer.wte.weight  # Эмбединги слов
        gpt_emb.wpe.weight = self.model.transformer.wpe.weight  # Эмбединги позиций
        gpt_emb.ln_f.weight = self.model.transformer.ln_f.weight  # Слой нормализации

        for n_layer in range(num_layers):
            gpt_emb.base_model.h[n_layer] = base_model.h[n_layer]  # Копирование слоев

        return gpt_emb

    def get_new_model(self, num_layers=3, name='gpt-embeder', save_path='gpt/model_emb'):
        """Создание модели на базе первых слоев gpt2 донора с перезаписью"""
        m = self._get_k_layer(num_layers, name)
        m.save_pretrained(save_path)
        return GPT2Model.from_pretrained(save_path)
