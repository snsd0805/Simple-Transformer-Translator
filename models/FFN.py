import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionwiseFeedForward(nn.Module):
    '''
        Position-size Feed Forward Network in Transformer block

        Args:
            dim (int): embedding in transformer block
            hidden (int): hidden state in this block
            dropout_rate (float): dropout layer's dropout rate in this block
        Inputs:
            x: (b, seq, dim)
        Outputs:
            x: (b, seq, dim)
    '''
    def __init__(self, dim, hidden, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(dim, hidden)
        self.linear2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
