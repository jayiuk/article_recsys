import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

class CNNlstm(nn.Module, Word2Vec):
    def __init__(self, input):
        super(CNNlstm, self).__init__()
        self.input = input
        self.layer_1 = nn.Sequential(
            nn.LSTM(input_size = 100, hidden_size = 5),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.Conv1d(in_channels = 100, out_channels = 50, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 1)
        )

   
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        return out
        