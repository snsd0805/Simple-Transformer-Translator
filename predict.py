import torch
import torch.nn as nn
import torchtext
import numpy
from utils.Vocab import get_vocabs
from utils.Dataset import TranslateDataset
from models.Transformer import Transformer
from models.Translator import TranslatorModel
import seaborn as sns
import matplotlib.pyplot as plt

BATCH_SIZE = 128
EPOCH_NUM = 100
LEARNING_RATE = 1e-4
ENGLISH_SEQ = 50
CHINESE_SEQ = 40
WORD_EMB_DIM = 100
DIM_IN_TRANSFORMER = 256
FFN_HIDDEN_DIM = 512
DEVICE = torch.device('cuda')
SHOW_NUM = 5
NUM_HEADS = 8
DROPOUT_RATE = 0.5

def en2tokens(en_sentence, en_vocab, for_model=False, en_seq=50):
    '''
        English to tokens

        Args:
            en_sentence (str)
            en_vocab (torchtext.Vocab)
            
            for_model (bool, default=False): if `True`, it will add <SOS>, <END>, <PAD> tokens
            en_seq (int): for padding <PAD>
        Outputs:
            tokens (LongTensor): (b,)
    '''
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    tokens = en_vocab( tokenizer(en_sentence.lower()) )
    if for_model:
        tokens = [ en_vocab['<SOS>'] ] + tokens + [ en_vocab['<END>'] ]
        tokens = tokens + [ en_vocab['<PAD>'] for _ in range(en_seq - len(tokens)) ]
    return torch.LongTensor(tokens)

def predict(en_str, model, en_vocab, ch_vocab):

    en_tokens = en2tokens(en_str, en_vocab, for_model=True, en_seq=ENGLISH_SEQ)
    en_tokens = en_tokens.unsqueeze(0).to(DEVICE)

    ch_tokens = torch.LongTensor([ ch_vocab['<PAD>'] for _ in range(CHINESE_SEQ) ]).unsqueeze(0).to(DEVICE)
    ch_tokens[0][0] = torch.tensor(ch_vocab['<SOS>'])

    model.eval()
    att = []
    with torch.no_grad():
        for index in range(0, CHINESE_SEQ):
            predict, score = model(en_tokens, ch_tokens)                    # b, seq, dim
            predict = torch.argmax(predict, dim=2)                          # b, seq
            att.append(score[0, :, index, :].unsqueeze(0))
            if index != (CHINESE_SEQ-1):
                ch_tokens[0][index+1] = predict[0][index]
    att = torch.cat(att, dim=0)                                             # seq, #head, ENGLISH_SEQ

    english_words = en_vocab.lookup_tokens(en_tokens[0].tolist())
    chinese_words = ch_vocab.lookup_tokens(ch_tokens[0].tolist())

    english_len, chinese_len = 0, 0
    for i in english_words:
        english_len += 1
        if i == '<END>':
            break
    for i in chinese_words:
        chinese_len += 1
        if i == '<END>':
            break

    return chinese_words, english_words, english_len, chinese_len, att

if __name__ == '__main__':
    # load tokenizer & vocabs
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    en_vocab, ch_vocab = get_vocabs()
    print("English Vocab size: {}\nChinese Vocab size: {}\n".format(len(en_vocab), len(ch_vocab)))

    # load model
    model = TranslatorModel(
        num_emb_en=len(en_vocab),
        num_emb_ch=len(ch_vocab),
        emb_dim=WORD_EMB_DIM,
        en_vocab=en_vocab,
        dim_in_transformer=DIM_IN_TRANSFORMER,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        en_seq=ENGLISH_SEQ,
        ch_seq=CHINESE_SEQ,
        device=DEVICE,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE
    ).to(DEVICE)

    model.load_state_dict(torch.load('translate_model.pth'))

    while(1):
        s = input("English: ")
        chinese_words, english_words, english_len, chinese_len, att = predict(s, model, en_vocab, ch_vocab)
        att = att.to('cpu').numpy()

        print("Chinese: {}".format(chinese_words))

        for i in range(8):
            mapping = att[:chinese_len, i, :english_len]

            plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
            plt.subplot(421+i)
            sns.heatmap(mapping, xticklabels=english_words[:english_len], yticklabels=chinese_words[:chinese_len])
        plt.show()