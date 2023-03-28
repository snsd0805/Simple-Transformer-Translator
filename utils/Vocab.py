import torchtext
import jieba
import logging
jieba.setLogLevel(logging.INFO)

def en_tokenizer_yeild(sentences):
    '''
        for building torchtext.vocab.Vocab (English)
        it use get_tokenizer() function to tokenizer English sentences
        then yield tokens to build_vocab_from_iterator() function to generate Vocab

        Args:
            sentences (list[str]): not case sensitive
    '''
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    for sentence in sentences:
        yield tokenizer(sentence.lower())

def ch_tokenizer_yeild(sentences):
    '''
        for building torchtext.vocab.Vocab (Chinese)
        it use jieba.cut function to tokenizer Chinese sentences
        then yield tokens to build_vocab_from_iterator() function to generate Vocab

        Args:
            sentences (list[str])
    '''

    for sentence in sentences:
        yield list(jieba.cut(sentence))

def generate_vocab(sentences, yield_f):
    '''
        Generate English or Chinese Vocab (torchtext.Vocab)

        Args:
            sentences (list[str]): English or Chinese sentences's list
            yield_f (function): en_tokenizer_yeild or ch_tokenizer_yeild, depends on which language's vocab to generate
        Outputs:
            vocab: (torchtext.Vocab)
    '''
    vocab = torchtext.vocab.build_vocab_from_iterator(
        yield_f(sentences),
        min_freq=1,
        special_first=True,
        specials=["<SOS>", "<END>", "<UNK>", "<PAD>"]
    )
    vocab.set_default_index(vocab['<UNK>'])
    return vocab

def get_vocabs():
    '''
        Generate English & Chinese two Vocab (torchtext.Vocab)

        Args: 
            None
        Outputs:
            en_vocab, ch_vocab: (torchtext.Vocab)
    '''
    with open('data/cmn_zh_tw.txt') as fp:
        sentences = fp.readlines()
    
    en_sentences, ch_sentences = [], []
    for index, line in enumerate(sentences):
        en, ch = line.replace('\n', '').split('\t')
        en_sentences.append( en.lower() )
        ch_sentences.append( ch )
    
    en_vocab = generate_vocab(en_sentences, en_tokenizer_yeild)
    ch_vocab = generate_vocab(ch_sentences, ch_tokenizer_yeild)
    return en_vocab, ch_vocab

