import torch
import torch.nn as nn
from models.Transformer import Transformer
import os

class TranslatorModel(nn.Module):
    '''
        Args:
            num_emb_en (int): length of vocab (how many words in vocab)
            num_emb_ch (int): length of vocab (how many words in vocab)
            emb_dim (int): word embedding dim (English's dim and Chineses' dim are same)
            en_vocab (torchtext.Vocab): for load Glove pretrained embedding
            dim_in_transformer (int)
            ffn_hidden_dim (int): for transformer's FFN module
            en_seq (int): English token' length
            ch_seq (int): Chinese token' length
            device (nn.Device)

            num_heads (int, default: 8)
            dropout_rate (float, default: 0.1)
        Inputs:
            en_tokens: (b, seq) LongTensor
            ch_tokens: (b, seq) LongTensor
        Outputs:
            x: (b, seq, num_emb_ch) probability
    '''
    def __init__(self, num_emb_en, num_emb_ch, emb_dim, en_vocab, dim_in_transformer, ffn_hidden_dim, en_seq, ch_seq, device, num_heads=8, dropout_rate=0.1):
        super(TranslatorModel, self).__init__()

        # load glove word embedding
        weight = self.get_glove_weight(en_vocab, num_emb_en, emb_dim)
        self.en_word_embedding = nn.Embedding(num_emb_en, emb_dim)
        self.en_word_embedding = self.en_word_embedding.from_pretrained(weight, freeze=True)

        # chinese word embedding
        self.ch_word_embedding = nn.Embedding(num_emb_ch, emb_dim)

        # transformer
        self.transformer = Transformer(
            emb_dim=emb_dim,
            dim=dim_in_transformer,
            ffn_hidden_dim=ffn_hidden_dim,
            encoder_seq=en_seq,
            decoder_seq=ch_seq,
            device=device,
            num_heads=8,
            dropout_rate=0.1
        )

        self.fc1 = nn.Linear(dim_in_transformer, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, num_emb_ch)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        

    def forward(self, en_tokens, ch_tokens):
        en_tokens = self.en_word_embedding(en_tokens)
        ch_tokens = self.ch_word_embedding(ch_tokens)
        x, score = self.transformer(en_tokens, ch_tokens)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, score

    def get_glove_weight(self, en_vocab, num_emb_en, emb_dim):
        '''
            Load embedding from GLOVE

            Args:
                en_vocab (torch.Vocab)
                num_emb_en (int): (how many word in vocab)
                emb_dim (int): word embedding's dim
        '''
        if os.path.isfile("data/word_embedding.pth"):
            weight = torch.load("data/word_embedding.pth")
        else:
            weight = torch.randn((num_emb_en, emb_dim))
            with open('data/glove.6B.100d.txt') as fp:
                lines = fp.readlines()
                for line in lines:
                    l = line.split(" ")
                    word = l[0]
                    emb  = l[1:]
                    emb = torch.tensor([ float(i) for i in emb ])
                    if word in en_vocab:
                        weight[en_vocab[word]] = emb
            torch.save(weight, "data/word_embedding.pth")
        return weight

