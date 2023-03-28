import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.TransformerEncoderBlock import TransformerEncoderBlock
from models.TransformerDecoderBlock import TransformerDecoderBlock
from models.TransformerEmbedding import TransformerEmbedding

class Transformer(nn.Module):
    '''
        Args:
            emb_dim (int): word embedding dim (input dim)
            dim (int): dim in transformer blocks
            ffn_hidden_dim (int): dim in FFN, bigger than dim
            encoder_seq (int): encoder input's length
            decoder_seq (int): decoder input's length
            device (nn.Device)
            num_heads (int, default=8)
            dropout_rate (float, default=0.1)
        Inputs:
            encoder_input: (b, encoder_seq, emb_dim)
            decoder_input: (b, decoder_seq, emb_dim)
        Outputs:
            decoder_output: (b, decoder_seq, dim)
    '''
    def __init__(self, emb_dim, dim, ffn_hidden_dim, encoder_seq, decoder_seq, device, num_heads=8, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.input_embedding = TransformerEmbedding(emb_dim, dim, encoder_seq, device)
        self.output_embedding = TransformerEmbedding(emb_dim, dim, decoder_seq, device)
        self.encoders = nn.ModuleList([
            TransformerEncoderBlock(dim, ffn_hidden_dim, encoder_seq, device, num_heads, dropout_rate) for _ in range(4)
        ])
        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(dim, ffn_hidden_dim, decoder_seq, device, num_heads, dropout_rate) for _ in range(4)
        ])
    
    def forward(self, encoder_input, decoder_input):
        encoder_input = self.input_embedding(encoder_input)
        decoder_input = self.output_embedding(decoder_input)
        for layer in self.encoders:
            encoder_input = layer(encoder_input)
        for layer in self.decoders:
            decoder_input, score = layer(decoder_input, encoder_input)
        return decoder_input, score

