from torch import nn


class NeuralNetwork(nn.Module):

    def __init__(self, input_dimension):
        super().__init__()
        self.linear = nn.Linear(input_dimension, 1)

    def forward(self, x):
        return self.linear(x) # Logits