import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.MultiHeadAttention import MultiHeadAttention
from models.FFN import PositionwiseFeedForward

class TransformerDecoderBlock(nn.Module):
    '''
        Args:
            dim (int): input embedding's output (emb_dim -> dim)
            ffn_hidden_dim (int): hidden state which in FFN block's dim
            seq (int): sequence's length
            device (nn.Device)
            num_heads (int, default: 8): for Multi-head Attention block
            dropout_rate (float, default: 0.1)
        Inputs:
            x: (b, seq, dim)
            enc: (b, encoder_seq, dim), encoder's output memory, encoder_seq is not arguments.
        Outputs:
            x: (b, seq, dim)
            score: (cross attention) (b, #heads, seq, seq)
    '''
    def __init__(self, dim, ffn_hidden_dim, seq, device, num_heads=8, dropout_rate=0.1):
        super(TransformerDecoderBlock, self).__init__()
        
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.layer_norm3 = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads=num_heads)
        self.ffn = PositionwiseFeedForward(dim, ffn_hidden_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

        self.seq = seq
        self.device = device
    
    def forward(self, x, enc):
        b = x.shape[0]

        # copy x, _x is original `x`
        _x = x.clone()

        # multi-head attention
        mask = self.getMask(b)                                                  # b, seq, seq
        x, score = self.attention(k=x, q=x, v=x, mask=mask)                     # b, seq, dim
        x = self.dropout(x)

        # Add && Norm
        x = _x + x
        x = self.layer_norm1(x)                                                 # b, seq, dim

        # copy x, _x is original `x`
        _x = x.clone()

        # mutl-head attention with encoder's memory
        x, score = self.attention(k=enc, q=x, v=enc)                            # b, seq, dim
        x = self.dropout(x)

        # Add && Norm
        x = _x + x
        x = self.layer_norm2(x)                                                 # b, seq, dim

        # copy x, _x is original `x`
        _x = x.clone()
        
        # FFN
        x = self.ffn(x)                                                         # b, seq, dim
        x = self.dropout(x)

        # Add && Norm
        x = _x + x
        x = self.layer_norm3(x)

        return x, score
    
    def getMask(self, batch_size):
        '''
            Return (b, seq, seq) mask
            0 1 1 1 1
            0 0 1 1 1
            0 0 0 1 1
            0 0 0 0 1
            0 0 0 0 0

            Inputs:
                batch_size (int)
        '''
        mask = torch.triu(torch.ones((self.seq, self.seq), dtype=torch.bool), diagonal=1)           # (seq, seq)
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)                                           # (b, seq, seq)
        mask = mask.to(self.device)
        return mask
