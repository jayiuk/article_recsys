import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init

class CNN(nn.Module):
    def __init__(self, input_channels, mid_channels, output_channels, kernel_size, stride, padding):
        super().__init__()
        self.input_channels = input_channels
        self.mid_channels = mid_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def conv1d_1(input_channels, output_channels, kernel_size):
        model = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size),
            nn.ReLU()
        )
        return model
    
    def conv1d_2(input_channels, mid_channels, output_channels, kernel_size, stride):
        model = nn.Sequential(
            nn.Conv1d(input_channels, mid_channels, kernel_size, stride),
            nn.ReLU(),
            nn.Conv1d(mid_channels, output_channels, kernel_size, stride),
            nn.ReLU()
        )
        return model
    
    def maxpool1d(input_channels, output_channels, kernel_size, stride, padding):
        pooling = nn.Sequential(
            nn.MaxPool1d(input_channels, output_channels, kernel_size, stride, padding),
            nn.ReLU()
        )
        return pooling
    
    def lstm():
    
    def forward(x):
        