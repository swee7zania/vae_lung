import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, layer_sizes, dropout, depth):
        super(MLP, self).__init__()
        layers = []
        for i in range(depth):
            in_dim = input_dim if i == 0 else layer_sizes[i - 1]
            out_dim = layer_sizes[i]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(layer_sizes[-1], 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
