import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init

class CNNlstm(nn.Module):
    def __init__(self):
        super(CNNlstm, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv1d(5, 3, 3),
            nn.ReLU(),
            nn.MaxPool1d(3, 1, 0)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv1d()
        )

   
    def forward(x, self):
        out = self.layer_1(x)
        out = self.layer_2(out)
        