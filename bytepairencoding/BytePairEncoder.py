import re
import os
import json
from collections import defaultdict
from time import time

class BytePairEncoder:
    def __init__(self, k: int, unk_token: str="<UNK>", encoder: dict=None):
        self.k = k
        self.unk_token = unk_token
        self.encoder = encoder
        if encoder is not None:
            self.vocab = set(encoder.keys())
        else:
            self.vocab = set()


    def train(self, words, freqs):
        ini = time()
        self.vocab = set()

        # Adding any single character:
        for word in words:
            for l in word.split():
                self.vocab.add(l)

        # Initializing encoder and decoder
        n = len(self.vocab)
        self.encoder = dict(zip(sorted(self.vocab), range(1, n+1)))
        self.decoder = dict(zip(range(1, n+1), sorted(self.vocab)))

        def encode_word(word):
            return ' '.join(map(lambda x: str(self.encoder.get(x)), word.split()))

        # Here we are going to store encoded words
        encoded_words = list(map(encode_word, words))

        # We are going to measure the frequency of any encoded bigram
        # e.g.
        # 20 30 40
        # 40 30 50
        # 20 30
        # 20 
        #     (20,30): 2
        #     (30,40): 1
        #     (40,30): 1
        #     (30,50): 1
        bigrams = defaultdict(int)
        pattern = r"(?=(?:\ |^)(\d+)\ (\d+)(?:\ |$))"
        matcher = re.compile(pattern)
        for i, encoded_word in enumerate(encoded_words):
            for m in matcher.findall(encoded_word):
                bigrams[m[0], m[1]] += freqs[i]


        for i in range(n+1, self.k+1):
            selected_bigram = max(bigrams, key=bigrams.get)

            content_bigram = self.decoder[int(selected_bigram[0])] + self.decoder[int(selected_bigram[1])]
            self.vocab.add(content_bigram)
            self.encoder[content_bigram] = i
            self.decoder[i] = content_bigram

            pattern = re.compile(r"(?<!\S)" + ' '.join(selected_bigram) + r"(?!\S)")
            for j in range(len(encoded_words)):
                encoded_words[j] = pattern.sub(str(i), encoded_words[j])

            bigrams = defaultdict(int)
            pattern = r"(?=(?:\ |^)(\d+)\ (\d+)(?:\ |$))"
            matcher = re.compile(pattern)
            for j, encoded_word in enumerate(encoded_words):
                for m in matcher.findall(encoded_word):
                    bigrams[m[0], m[1]] += freqs[j]
        
        print(f"Trained {self.k} tokens in {time()-ini: .2f}s")


    def encode(self, word):
        # Greedy encoding
        n = len(word)
        i, j = 0, n

        encoded_word = []
        while (word[i:n] not in self.encoder) and ((n-1)>i):
            j = n - 1
            while (word[i:j] not in self.encoder) and (j>i):
                j -= 1
            encoded_word.append(self.encoder.get(word[i:j]))
            if i<j:
                i = j
            else:
                i += 1
        encoded_word.append(self.encoder.get(word[i:n]))

        return encoded_word


    def decode(self, encoded_word, sep=''):
        word = sep.join(self.decoder.get(code) for code in encoded_word)

        return word

    
    def save(self, dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
        with open(os.path.join(dirname, "encoder.json"), 'w') as out:
            json.dump(self.encoder, out)
        
        with open(os.path.join(dirname, "decoder.json"), 'w') as out:
            json.dump(self.decoder, out)

    def load(self, dirname):
        with open(os.path.join(dirname, "encoder.json"), 'r') as input:
            self.encoder = json.loads(input)

        with open(os.path.join(dirname, "decoder.json"), 'r') as input:
            self.decoder = json.loads(input)
        
        self.k = len(self.encoder)
        self.vocab = set(self.encoder.keys())

