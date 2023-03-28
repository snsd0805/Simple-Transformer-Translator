import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.MultiHeadAttention import MultiHeadAttention
from models.FFN import PositionwiseFeedForward

class TransformerEncoderBlock(nn.Module):
    '''
        Args:
            dim (int): input embedding's output (emb_dim -> dim)
            ffn_hidden_dim (int): hidden state which in FFN block's dim
            seq (int): sequence's length
            device (nn.Device)
            num_heads (int, default: 8): for Multi-head Attention block
            dropout_rate (float, default=0.1)
        Inputs:
            x: (b, seq, dim)
        Outputs:
            x: (b, seq, dim)

    '''
    def __init__(self, dim, ffn_hidden_dim, seq, device, num_heads=8, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads=num_heads)
        self.ffn = PositionwiseFeedForward(dim, ffn_hidden_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

        self.seq = seq
        self.device = device
    
    def forward(self, x):

        # _x is original `x`
        _x = x.clone()

        # multi-head attention
        x, score = self.attention(k=_x, q=_x, v=_x, mask=None)                  # b, seq, dim
        x = self.dropout(x)

        # Add && Norm
        x = _x + x
        x = self.layer_norm1(x)                                                 # b, seq, dim

        # copy x, _x is original `x`
        _x = x.clone()

        # FFN
        x = self.ffn(x)                                                         # b, seq, dim
        x = self.dropout(x)

        # Add && Norm
        x = _x + x
        x = self.layer_norm2(x)

        return x
