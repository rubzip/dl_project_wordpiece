import string
from collections import defaultdict
import re
import random
import pandas as pd


def load_sst_data(path,
                  easy_label_map={0:0, 1:0, 2:None, 3:1, 4:1}):
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            example = {}
            example['label'] = easy_label_map[int(line[1])]
            if example['label'] is None:
                continue

            # Strip out the parse information and the phrase labels--
            # ---we don't need those here
            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            example['text'] = text[1:]
            data.append(example)
    random.seed(1)
    random.shuffle(data)
    return data

def clean(word: str, sep=''):
    return sep.join(filter(lambda x: x in string.ascii_lowercase, word.lower()))

def tokenize(sentence: str):
    return sentence.lower().split()

def count_freq(dataset: dict, sep):
    word_freq = defaultdict(lambda: 0)
    for tag_sent in dataset:
        for word in tokenize(tag_sent["text"]):
            clean_word = clean(word, sep)
            if clean_word != '':
                word_freq[clean_word] += 1
    return word_freq

def word_freq_dictionary(dataset: dict, sep=' '):
    word_freq = count_freq(dataset, sep)
    df = pd.DataFrame.from_dict(word_freq, orient='index').reset_index()
    df.columns = ["word", "freq"]
    df.sort_values("freq", ascending=False, inplace=True)
    return df

class Tokenizer:
    def __init__(self, k: int, unk_token: str="<UNK>", encoder: dict=None):
        self.k = k
        self.unk_token = unk_token
        self.encoder = encoder
        if encoder is not None:
            self.vocab = set(encoder.keys())
        else:
            self.vocab = set()

    def train(self, word_freq):
        words = word_freq.sort_values("freq", ascending=False).head(self.k)["word"]
        
        self.vocab = set(words)
        self.encoder = dict(zip(sorted(words), range(1, self.k+1)))
        self.decoder = dict(zip(range(1, self.k+1), sorted(words)))
        self.encoder[self.unk_token] = 0
        self.decoder[0] = self.unk_token

    def encode(self, word):
        clean_word = clean(word.lower(), '')
        return self.encoder.get(clean_word, 0)


    def decode(self, encoded_word):
        word = self.decoder.get(encoded_word, self.unk_token)
        return word

    
    def save(self, dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
        with open(os.path.join(dirname, "tokenizer_encoder.json"), 'w') as out:
            json.dump(self.encoder, out)
        
        with open(os.path.join(dirname, "tokenizer_decoder.json"), 'w') as out:
            json.dump(self.decoder, out)

    def load(self, dirname):
        with open(os.path.join(dirname, "tokenizer_encoder.json"), 'r') as input:
            self.encoder = json.loads(input)

        with open(os.path.join(dirname, "tokenizer_decoder.json"), 'r') as input:
            self.decoder = json.loads(input)
        
        self.k = len(self.encoder)
        self.vocab = set(self.encoder.keys())