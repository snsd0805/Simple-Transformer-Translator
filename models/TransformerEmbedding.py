import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.PositionEncode import PositionEncode

class TransformerEmbedding(nn.Module):
    def __init__(self, emb_dim, dim, seq, device, dropout_rate=0.1):
        super(TransformerEmbedding, self).__init__()
        self.position_encoding = PositionEncode(dim, device)
        self.input_embedding = nn.Linear(emb_dim, dim)

        self.dim = dim
        self.seq = seq
        self.device = device

    def forward(self, x):
        b = x.shape[0]
        x = self.input_embedding(x)
        position_emb = self.getPositionEncoding(b)                              # b, seq, dim
        x  += position_emb
        return x
    
    def getPositionEncoding(self, batch_size):
        '''
            Return (b, seq, dim) position encode

            Inputs:
                batch_size (int)
        '''
        time_seq = torch.LongTensor(range(self.seq)).to(self.device)
        emb = self.position_encoding(time_seq)                                  # (seq, dim)
        emb = emb[:, :, None].permute(2, 0, 1).repeat(batch_size, 1, 1)    # (batch_size, seq, dim)
        return emb
