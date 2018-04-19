import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

in_seqs = [
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9]
]
input_seq = np.zeros((8, 3, 10), dtype="float32")

out_seqs = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9]
]
output_seq = np.zeros((8, 2, 9), dtype="float32")

for i, s in enumerate(in_seqs):
    for j, t in enumerate(s):
        input_seq[i, j, t] = 1.

for i, s in enumerate(out_seqs):
    for j, t in enumerate(s):
        output_seq[i, j, t - 1] = 1.

output_seq = np.zeros((8, 9), dtype="float32")
output_seq[0, 0] = 1.
output_seq[2, 0] = 1.
output_seq[6, 0] = 1.

model = Sequential()
model.add(LSTM(256, input_shape=(3, 10)))
model.add(Dense(9, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
print(model.summary())
model.fit(input_seq, output_seq, batch_size=64, epochs=25)

for i in range(3):
    print(model.predict(input_seq[i:i+1]))