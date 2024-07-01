import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init

class CNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size

    def conv1d_1(input_channels, output_channels, kernel_size):
        model = nn.Sequentila(
            nn.Conv1d(input_channels, output_channels, kernel_size)
            nn.ReLU()
        )