import torch
from torch.utils.data import Dataset
import random
import jieba

class TranslateDataset(Dataset):
    '''
        Generate en, ch tokens (numeric token)

        Args:
            random_seed (int)
            tokenizer (torchtext tokenizer): English
            en_vocab (torchtext.Vocab): English ver
            ch_vocab (torchtext.Vocab): Chinese ver
            en_seq (int): english token's length (it will padding to this length)
            ch_seq (int): chinese token's length (it will padding to this length)
            train_ratio (float, default: 0.8)
            val (bool, default: False)
    '''
    def __init__(self, random_seed, tokenizer, en_vocab, ch_vocab, en_seq, ch_seq, train_ratio=0.8, val=False):
        super(Dataset, self).__init__()
        random.seed(random_seed)

        self.en_vocab = en_vocab
        self.ch_vocab = ch_vocab
        
        # read file
        with open('data/cmn_zh_tw.txt') as fp:
            lines = fp.readlines()
            length = len(lines)
        
        # random & split
        random.shuffle(lines)
        if val:
            self.data = lines[ int(length*train_ratio): ]
        else:
            self.data = lines[ :int(length*train_ratio) ]
        
        # tokenizer
        self.en_data, self.ch_data = [], []
        for index, line in enumerate(self.data):
            en, ch = line.replace('\n', '').split('\t')
            
            en_tokens = en_vocab(tokenizer(en.lower()))
            en_tokens = [ en_vocab['<SOS>'] ] + en_tokens + [ en_vocab['<END>'] ]
            en_tokens = en_tokens + [ en_vocab['<PAD>'] for _ in range(en_seq - len(en_tokens)) ]
            self.en_data.append(en_tokens)

            ch_tokens = ch_vocab(list(jieba.cut(ch)))
            ch_tokens = [ ch_vocab['<SOS>'] ] + ch_tokens + [ ch_vocab['<END>'] ]
            ch_tokens = ch_tokens + [ ch_vocab['<PAD>'] for _ in range(ch_seq - len(ch_tokens)) ]
            self.ch_data.append(ch_tokens)
        
    def __len__(self):
        return len(self.en_data)
    
    def __getitem__(self, index):
        target = torch.LongTensor( self.ch_data[index][1:] + [ self.ch_vocab['<PAD>'] ] )
        return (torch.LongTensor(self.en_data[index]), torch.LongTensor(self.ch_data[index])), target