from WordPiece import WordPiece

words_raw = [
    "huggingface",
    "hugging",
    "face",
    "hug",
    "hugger",
    "learning",
    "learner",
    "learners",
    "learn",
]

freqs = [
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    1,
]

def transform(word):
    return ' '.join(l if i==0 else "##" + l for i, l in enumerate(word))

words = list(map(transform, words_raw))
wp = WordPiece(25)
wp.train(words, freqs)

print(wp.encoder)