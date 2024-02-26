from BytePairEncoder import BytePairEncoder

words = ["l o w _", "l o w e s t _", "n e w e r _", "w i d e r _", "n e w _"]
freqs = [5, 2, 6, 3, 2]

bpe = BytePairEncoder(19)
bpe.train(words, freqs)

print(bpe.encoder)