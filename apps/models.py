"""Shared small encoder for the optimizer and compression experiments."""
from needle import nn


class SmallEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, feature_dim=64,
                 num_classes=10, device=None, extra_hidden_layers=0):
        super().__init__()
        if type(extra_hidden_layers) is not int or extra_hidden_layers < 0:
            raise ValueError('Extra hidden layer count must be a nonnegative integer')
        self.extra_hidden_layers = extra_hidden_layers
        self.input = nn.Linear(input_dim, hidden_dim, device=device)
        self.extra = [nn.Linear(hidden_dim, hidden_dim, device=device)
                      for _ in range(extra_hidden_layers)]
        self.hidden = nn.Linear(hidden_dim, feature_dim, device=device)
        self.classifier = nn.Linear(feature_dim, num_classes, device=device)
        self.activation = nn.ReLU()

    def features(self, x):
        x = self.activation(self.input(x))
        for layer in self.extra:
            x = self.activation(layer(x))
        return self.activation(self.hidden(x))

    def muon_parameters(self):
        return [layer.weight for layer in self.extra] + [self.hidden.weight]

    def forward(self, x):
        return self.classifier(self.features(x))
