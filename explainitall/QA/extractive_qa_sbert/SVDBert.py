import torch.nn as nn
from transformers import BertModel


class SVDLinearLayer(nn.Module):
    """
    Линейный слой, использующий пространство уменьшенной размерности, использующий SVD.
    """

    def __init__(self, in_features, out_features, h_dim, bias=True):
        super(SVDLinearLayer, self).__init__()
        self.encoder = nn.Linear(in_features, h_dim, bias=False)
        self.decoder = nn.Linear(h_dim, out_features, bias=bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SVDBertModel(BertModel):
    """
    Модель BERT с уменьшенной размерностью в определенных слоях с использованием подхода SVD.
    """

    def __init__(self, config, svd_dim=5):
        super(SVDBertModel, self).__init__(config)

        for i, layer in enumerate(self.encoder.layer):
            intermediate_size = layer.intermediate.dense.out_features
            output_size = layer.output.dense.out_features

            if i > 0:
                layer.intermediate.dense = SVDLinearLayer(
                    layer.intermediate.dense.in_features,
                    intermediate_size,
                    svd_dim
                )
                layer.output.dense = SVDLinearLayer(
                    layer.output.dense.in_features,
                    output_size,
                    svd_dim
                )
            else:
                layer.intermediate.dense = nn.Linear(
                    layer.intermediate.dense.in_features,
                    intermediate_size,
                    bias=True
                )
                layer.output.dense = nn.Linear(
                    layer.output.dense.in_features,
                    output_size,
                    bias=True
                )
