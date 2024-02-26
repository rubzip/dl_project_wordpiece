import pandas as pd
from BytePairEncoder import BytePairEncoder

words_freqs = pd.read_csv("./words.csv")
words = words_freqs["words"]
freqs = words_freqs["freqs"]

bpe = BytePairEncoder(5000)
bpe.train(words, freqs)
 
print(bpe.encoder)