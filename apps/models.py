"""Shared small encoder for the optimizer and compression experiments."""
from needle import nn


class SmallEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, feature_dim=64,
                 num_classes=10, device=None):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim, device=device)
        self.hidden = nn.Linear(hidden_dim, feature_dim, device=device)
        self.classifier = nn.Linear(feature_dim, num_classes, device=device)
        self.activation = nn.ReLU()

    def features(self, x):
        x = self.activation(self.input(x))
        return self.activation(self.hidden(x))

    def forward(self, x):
        return self.classifier(self.features(x))
