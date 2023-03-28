import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    '''
        Multi-Head Self Attention Block

        Args:
            dim (int): input & output dim
            num_heads (int, default=8): number of heads
        Inputs:
            k: (b, seq, dim), it's not key value from anywhere, it's an embedding ready to get into W_k
            q: (b, seq, dim), like k
            v: (b, seq, dim), like v
            mask (default None): BoolTensor, (b, seq, dim)
        Outputs:
            ans: (b, seq, dim)
            score: (b, #heads, seq, seq) attention score which after softmax
    '''
    def __init__(self, dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.wk = nn.Linear(dim, dim)                                       # b, seq, dim
        self.wq = nn.Linear(dim, dim)                                       # b, seq, dim
        self.wv = nn.Linear(dim, dim)                                       # b, seq, dim
        self.fc = nn.Linear(dim, dim)

    def forward(self, k, q, v, mask=None):
        b, seq, dim = k.shape
        k = self.wk(k)                                                      # b, seq, dim
        q = self.wq(q)                                                      # b, seq, dim
        v = self.wv(v)                                                      # b, seq, dim

        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)   # b, #heads, seq, #head_dim
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)   # b, #heads, seq, #head_dim
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)   # b, #heads, seq, #head_dim

        k = k.transpose(2, 3)                                               # b, #heads, #head_dim, seq

        score = torch.matmul(q, k) / (math.sqrt(self.head_dim))             # b, #heads, seq, seq
        if mask != None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            score = score.masked_fill(mask, value=torch.tensor(-(1e20)))
            # print(score[0][0][2])
            # for i in score[0][0]:
            #     print(i)
        score = F.softmax(score, dim=-1)

        ans = torch.matmul(score, v)                                        # b, #heads, seq, head_dim

        ans = ans.transpose(1, 2).reshape((b, -1, dim))                     # b, seq, dim
        ans = self.fc(ans)                                                  # b, seq, dim

        return ans, score
