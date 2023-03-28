import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionEncode(nn.Module):
    '''
        Args:
            emb_dim (int): position embedding dim
            device (nn.device)
        Inputs:
            time_seq: LongTensor (b, )
    '''
    def __init__(self, emb_dim, device):
        super(PositionEncode, self).__init__()
        seq = torch.tensor([ i//2 for i in range(emb_dim) ]) / emb_dim
        self.base = 1/torch.pow(10000, seq).to(device)                           # (dim, )
        self.emb_dim = emb_dim

    def forward(self, time_seq):
        b = time_seq.shape[0]
        base = self.base[:, None].reshape(1, -1).repeat(b, 1)                   # (b, dim)
        time_seq = time_seq[:, None]
        # .repeat(1, self.emb_dim)                    # (b, dim)


        ans = base * time_seq                                                   # (b, dim)
        ans[:, 0::2] = torch.sin(ans[:, 0::2])
        ans[:, 1::2] = torch.cos(ans[:, 1::2])

        return ans
