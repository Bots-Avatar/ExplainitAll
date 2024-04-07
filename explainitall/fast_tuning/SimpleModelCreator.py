import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model

from .Embedder import GPTEmbedder
from .trainers.DenceKerasTrainer import GPTFastTrainer


def get_dataset_dense(txts, embedder: GPTEmbedder, tokenizer: GPT2Tokenizer, n_layer_index='all'):
    """
    Создает датасет из текстов с помощью заданного embedder и tokenizer.
    """
    list_x, list_y = [], []
    for txt in txts:
        words = txt.split(' ')
        for i in range(0, len(words), 25):
            text = ' '.join(words[i:])
            emb = embedder.get_embs_from_gpt(text, n_layer_index=n_layer_index)[:-1][:1024]
            ids = np.array(tokenizer(text)['input_ids'])[1:]
            list_x.append(emb)
            list_y.append(ids)

    return np.concatenate(list_x), np.concatenate(list_y)


def gpt_build(trainer: GPTFastTrainer, gpt_emb: GPT2Model, tokenizer: GPT2Tokenizer, y_set,
              path_to_save='gpt_model_new'):
    """
    Создает и сохраняет новую модель GPT.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    gpt_emb.to(device)

    w_out_matr = trainer.adapter_layer.get_weights()[0] @ trainer.keras_out_weight.T
    config_gpt = gpt_emb.config
    new_model = GPT2LMHeadModel(config_gpt)
    new_model.transformer = gpt_emb
    new_model.lm_head = nn.Linear(config_gpt.n_embd, config_gpt.vocab_size, bias=False)
    new_model.lm_head.weight = nn.Parameter(torch.tensor(w_out_matr, device=device))

    new_model.save_pretrained(path_to_save)
    tokenizer.save_pretrained(path_to_save)
    np.save(f'{path_to_save}/set.data', np.array(y_set))


class SimpleCreator:
    def __init__(self, model: GPT2Model, tokenizer: GPT2Tokenizer):
        self.tokenizer = tokenizer
        main_embedder = GPTEmbedder(tokenizer, model)
        self.gpt_emb = main_embedder.get_new_model(num_layers=-1)
        self.cut_embedder = GPTEmbedder(self.tokenizer, self.gpt_emb)
        self.trainer = GPTFastTrainer(model)

    def train(self, data, lr=0.0003, bs=64, epochs=6, val_split=0.0, save_path='new_model'):
        x, y = get_dataset_dense(data, self.cut_embedder, self.tokenizer)
        net = self.trainer.create_net()
        self.trainer.train(net, x, y, lr=lr, bs=bs, epochs=epochs, val_split=val_split)
        y_set = list(set(y))
        gpt_build(self.trainer, self.gpt_emb, self.tokenizer, y_set, save_path)
        return net
