import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

MAX_WORDS = 5000
EMBED_DIM = 200

with open('meditations.txt') as f:
    lines = f.read().split('\n')

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(lines)
word_index = tokenizer.word_index


embeddings_index = {}
with open('glove.6B.200d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


embedding_matrix = np.zeros((len(word_index) + 1, EMBED_DIM))
missing_count = 0
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:
        missing_count += 1
    else:
        embedding_matrix[i] = embedding_vector

print('%s missing embeddings.' % missing_count)

np.save('matrix.npy', embedding_matrix)
