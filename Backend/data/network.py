import os

import numpy as np
import torch
from torch import nn

from Backend.data_manager.data_manager import data_manager


class NeuralNetwork(nn.Module):

    def __init__(self, input_dimension):
        super().__init__()
        self.linear = nn.Linear(input_dimension, 1)

    def forward(self, x):
        return self.linear(x)