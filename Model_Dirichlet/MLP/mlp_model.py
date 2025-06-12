import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, latent_size, base, layer_sizes, dropout, depth):
        super(MLP, self).__init__()

        layers = []
        size1, size2, size3 = layer_sizes

        if depth == 4:
            layers = [
                nn.Linear(latent_size * base, size1),
                nn.GELU(),
                nn.BatchNorm1d(size1),
                nn.Dropout(dropout),

                nn.Linear(size1, size2),
                nn.GELU(),
                nn.BatchNorm1d(size2),
                nn.Dropout(dropout),

                nn.Linear(size2, size3),
                nn.GELU(),
                nn.BatchNorm1d(size3),
                nn.Dropout(dropout),

                nn.Linear(size3, 1),
                nn.Sigmoid()
            ]

        elif depth == 5:
            layers = [
                nn.Linear(latent_size * base, size1),
                nn.GELU(),
                nn.BatchNorm1d(size1),
                nn.Dropout(dropout),

                nn.Linear(size1, size2),
                nn.GELU(),
                nn.BatchNorm1d(size2),
                nn.Dropout(dropout),

                nn.Linear(size2, size2),
                nn.GELU(),
                nn.BatchNorm1d(size2),
                nn.Dropout(dropout),

                nn.Linear(size2, size3),
                nn.GELU(),
                nn.BatchNorm1d(size3),
                nn.Dropout(dropout),

                nn.Linear(size3, 1),
                nn.Sigmoid()
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
