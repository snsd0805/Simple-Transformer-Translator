from torchinfo import summary
import torchtext
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy
from utils.Vocab import get_vocabs
from utils.Dataset import TranslateDataset
from models.Transformer import Transformer
from models.Translator import TranslatorModel

def train():
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
    DROPOUT_RATE = 0.3

    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    en_vocab, ch_vocab = get_vocabs()
    print("English Vocab size: {}\nChinese Vocab size: {}\n".format(len(en_vocab), len(ch_vocab)))

    train_set = TranslateDataset(10, tokenizer, en_vocab, ch_vocab, ENGLISH_SEQ, CHINESE_SEQ)
    val_set   = TranslateDataset(10, tokenizer, en_vocab, ch_vocab, ENGLISH_SEQ, CHINESE_SEQ, val=True)
    print("Train set: {}\nValid set: {}\n".format(len(train_set), len(val_set)))


    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    min_val_loss = 99
    for epoch in range(EPOCH_NUM):
        model.train()

        loss_sum = {'train': 0, 'val': 0}
        acc_sum = {'train': 0, 'val': 0}
        count = {'train': 0, 'val': 0}
        for (en_tokens, ch_tokens), y in train_loader:
            b = len(y)
            count['train'] += b

            en_tokens, ch_tokens = en_tokens.to(DEVICE), ch_tokens.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(en_tokens, ch_tokens)                                # (b, seq, dim)
            prediction = prediction.view(-1, len(ch_vocab))                         # (b*seq, dim)
            y = y.view(-1)                                                          # (b*seq)

            loss = criterion(prediction, y)
            loss_sum['train'] += loss.item()
            # print(loss.item())

            loss.backward()
            optimizer.step()

            acc_sum['train'] += (torch.argmax(prediction, dim=1) == y).sum()

        prediction = torch.argmax(prediction, dim=1).view(b, -1)
        for index, seq in enumerate(prediction):
            print(en_vocab.lookup_tokens(en_tokens[index].tolist()))
            print(ch_vocab.lookup_tokens(seq.tolist()))
            print()
            if index >= SHOW_NUM:
                break
            
        # val
        model.eval()
        with torch.no_grad():
            for (en_tokens, ch_tokens), y in val_loader:
                b = len(y)
                count['val'] += b

                en_tokens, ch_tokens = en_tokens.to(DEVICE), ch_tokens.to(DEVICE)
                y = y.to(DEVICE)

                prediction = model(en_tokens, ch_tokens)                                # (b, seq, dim)
                prediction = prediction.view(-1, len(ch_vocab))                         # (b*seq, dim)
                y = y.view(-1)                                                          # (b*seq)

                loss = criterion(prediction, y)
                loss_sum['val'] += loss.item()
                # print(loss.item())

                acc_sum['val'] += (torch.argmax(prediction, dim=1) == y).sum()

            prediction = torch.argmax(prediction, dim=1).view(b, -1)
            for index, seq in enumerate(prediction):
                print(en_vocab.lookup_tokens(en_tokens[index].tolist()))
                print(ch_vocab.lookup_tokens(seq.tolist()))
                print()
                if index >= SHOW_NUM:
                    break
        print(count)
        print(loss_sum)
        print("EPOCH {}: with lr={}, loss: {} acc: {}".format(epoch, LEARNING_RATE, loss_sum['train']/len(train_loader), acc_sum['train']/count['train']/CHINESE_SEQ))
        print("EPOCH {}: with lr={}, loss: {} acc: {} (val)".format(epoch, LEARNING_RATE, loss_sum['val']/len(val_loader), acc_sum['val']/count['val']/CHINESE_SEQ))
        if((loss_sum['val']/len(val_loader)) < min_val_loss):
            min_val_loss = loss_sum['val']/len(val_loader)
            torch.save(model.state_dict(), 'translate_model.pth')
        print("MIN: {}".format(min_val_loss))
        print()
train()
