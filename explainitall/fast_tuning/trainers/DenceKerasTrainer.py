import torch
import torch.nn as nn
import torch.optim as optim


class GPTFastTrainer(nn.Module):
    def __init__(self, gpt_model):
        super(GPTFastTrainer, self).__init__()
        self.inp_dim = gpt_model.config.n_embd
        self.outp_dim = gpt_model.config.vocab_size

        # Адаптирующий слой
        self.adapter_layer = nn.Linear(self.inp_dim, self.inp_dim, bias=False)

        # Выходной слой
        self.out_layer = nn.Linear(self.inp_dim, self.outp_dim, bias=True)
        self.out_layer.weight.requires_grad = False  # Замораживание весов выходного слоя

        # Инициализация весов
        with torch.no_grad():
            self.out_layer.weight.copy_(gpt_model.lm_head.weight.detach().transpose(0, 1))
            self.adapter_layer.weight.fill_(1)  # Единичная матрица

    def forward(self, x):
        x = self.adapter_layer(x)
        x = self.out_layer(x)
        return torch.softmax(x, dim=-1)

    def train_model(self, net, x, y, lr=0.0003, bs=64, epochs=3, val_split=0.0):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adamax(net.parameters(), lr=lr)

        # Разделение данных на обучающую и валидационную выборки
        split_idx = int(len(x) * (1 - val_split))
        train_x, val_x = x[:split_idx], x[split_idx:]
        train_y, val_y = y[:split_idx], y[split_idx:]

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = net(train_x)
            loss = criterion(outputs, train_y)
            loss.backward()
            optimizer.step()

            if val_split > 0:
                val_outputs = net(val_x)
                val_loss = criterion(val_outputs, val_y)
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")
            else:
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
