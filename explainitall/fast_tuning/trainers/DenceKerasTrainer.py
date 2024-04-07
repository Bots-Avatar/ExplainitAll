import keras.layers as L
import numpy as np
import torch
from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras as K


class GPTFastTrainer:
    """ Создание обучаемого последнего слоя"""

    def __init__(self, gpt_model):
        self.inp_dim = gpt_model.config.n_embd
        self.outp_dim = gpt_model.config.vocab_size

        if torch.cuda.is_available():
            gpt_model.to('cpu')

        self.keras_out_weight = gpt_model.lm_head.weight.detach().numpy().transpose()  # Получение весовых коэффициентов
        self.keras_adapter_weight = np.eye(self.inp_dim, dtype=float)  # Единичная матрица (коэф. адаптирующего слоя)

        self.adapter_layer = Dense(self.inp_dim, use_bias=False, activation='linear')  # Адаптирующий слой
        self.out_layer = Dense(use_bias=True, units=self.outp_dim, activation='linear',
                               trainable=False)  # Выходной слой (необучается)

        if torch.cuda.is_available():
            gpt_model.to('cuda:0')

    # Создание сети для тюнинга #
    def creat_net(self):
        net = Sequential()
        net.add(L.Input(self.inp_dim))
        net.add(self.adapter_layer)
        net.add(self.out_layer)
        net.add(L.Activation(activation='softmax'))
        net.compile()
        # Загрузка весов в выходной слой
        self.out_layer.set_weights([self.keras_out_weight, np.zeros(self.outp_dim)])
        # Загрузка весов в слой адаптера
        self.adapter_layer.set_weights([self.keras_adapter_weight])
        return net

    def set_variety_of_answers(self, y, variety=0, min_prob=1e-300):
        """Пересоздание слоя с установкой вариативности генерации"""
        set_tokens = set(y)
        bias = np.zeros(self.outp_dim)
        coef_mask = np.log2(variety + min_prob) / np.log2(np.e)
        bias += coef_mask

        for token in set_tokens:
            bias[token] = 0

        self.out_layer.set_weights([self.keras_out_weight, bias])

    def train(self, net, x, y, lr=0.0003, bs=64, epochs=3, val_split=0.0):
        """Обучение сети"""
        self.set_variety_of_answers(y)
        opt = K.optimizers.Adamax(learning_rate=lr)
        net.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        net.fit(x, y, batch_size=bs, epochs=epochs, validation_split=val_split)
