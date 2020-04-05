import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU, LSTM, TimeDistributed 
from tensorflow.keras.callbacks import ModelCheckpoint

MAX_WORDS = 5000
SEQ_LEN = 10
EPOCHS = 10
UNITS = 500
EMBED_DIM = 200

with open('meditations.txt') as f:
    lines = f.read().split('\n')

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

x = []
y = []

for s in sequences:
    i = 0
    while i + SEQ_LEN < len(s):
        x.append(s[i:i+SEQ_LEN])
        y.append(s[i+1:i+SEQ_LEN+1])
        i += 1

embedding_matrix = np.load('matrix.npy')
vocab_size = embedding_matrix.shape[0]

model = Sequential()
model.add(Embedding(vocab_size, EMBED_DIM, weights=[embedding_matrix], input_length=SEQ_LEN, batch_input_shape=[1, None], trainable=False))
model.add(GRU(UNITS, return_sequences=True, stateful=True, dropout=0.1))
model.add(Dense(vocab_size))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
print(model.summary())


checkpointer = ModelCheckpoint(
    filepath='weights-{epoch:02d}.hdf5',
    save_weights_only=True
)

model.fit(x, y, epochs=EPOCHS, callbacks=[checkpointer])

