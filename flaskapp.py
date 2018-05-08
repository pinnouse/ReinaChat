from io import open
import os.path
from configparser import ConfigParser
import vocab_builder
here = os.path.dirname(__file__)
config = ConfigParser()
config.read(os.path.join(here, 'bot-config.ini'))
batch_size = int(config['DEFAULT']['batch_size'])
epochs = int(config['DEFAULT']['epochs'])
latent_dim = int(config['DEFAULT']['latent_dim'])
num_samples = int(config['DEFAULT']['num_samples'])
data_path = config['DEFAULT']['data_path']
vocab_size = int(config['DEFAULT']['vocab_size'])
max_seq_len = int(config['DEFAULT']['max_seq_len'])

input_token_index = dict()
target_token_index = dict()
num_encoder_tokens = 0
num_decoder_tokens = 0
with open(os.path.join(here, data_path + 'in.vocab'), 'r', encoding='utf-8', errors='ignore') as f:
    for i, row in enumerate(f):
        input_token_index[row] = i
        
with open(os.path.join(here, data_path + 'tg.vocab'), 'r', encoding='utf-8', errors='ignore') as f:
    for i, row in enumerate(f):
        target_token_index[row] = i

num_encoder_tokens = len(input_token_index) - 1
num_decoder_tokens = len(target_token_index) - 1

from keras import Model
from keras.layers import Input, LSTM, Dense

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _, = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

data = (max_seq_len, num_samples, epochs, batch_size, latent_dim, vocab_size)
model_location = os.path.join(here, "model/bot-%d %dsamples (%d-%d-%d-%d).h5" % data)
model.load_weights(model_location)
model._make_predict_function()
model.summary()

encoder_model = Model(encoder_inputs, encoder_states)
encoder_model._make_predict_function()

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)
decoder_model._make_predict_function()

import numpy as np
import random
from nltk.tokenize import RegexpTokenizer

def sample(a, randomness=1):
    # randomness is how many other words may be possible
    a = np.array(a)
    max_score_indeces = a.argsort()[-(1+randomness):][::-1]
    sorted_weights = []
    for i in max_score_indeces:
        sorted_weights.append((i, a[i]))
    sorted_weights, sorted_scores = zip(*sorted(sorted_weights, key=lambda x:float(x[1]), reverse=True))
    for i in range(len(sorted_scores) - 1):
        if i > 0:
            if sorted_scores[i] * 0.90 > sorted_scores[i]: #if the next score is not within 90% of the initial one, cut the scoring
                sorted_weights = sorted_weights[:i]
                break
    # print(list(sorted_weights))
    for i in list(sorted_weights):
        if random.random() < 0.8:
            return i
    return sorted_weights[0]

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["<GO>"]] = 1.

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token_index = sample(output_tokens[0, -1, :], 2)
        sampled_w = list(target_token_index.keys())[sampled_token_index]
        # print("sampled token index:", sampled_token_index, "word:", sampled_w)
        decoded_sentence += sampled_w + " "

        if sampled_w == "<EOS>" or len(decoded_sentence.split(' ')) > max_seq_len:
            stop_condition = True
        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]
    
    return decoded_sentence

tokenizer = RegexpTokenizer(r'\w+|\<[A-Z]+\>|\$[a-z]+|&[a-z]+;|[a-z]?\'[a-z]+|[.?!]') # Special commands are: $command
def sentence_to_seq(sentence):
    sentence = tokenizer.tokenize(sentence)
    seq = np.zeros((1, max_seq_len, num_encoder_tokens), dtype='float32')
    
    read_sentence = ""
    for i in range(min(max_seq_len, len(sentence))):
        w = sentence[i].lower()
        # print(w)
        if w not in list(input_token_index.keys()):
            w = "<UNK>"
        seq[0, i, input_token_index[w]] = 1.
        read_sentence += w + " "
        
    print("Read sentence: ", read_sentence)

    return (seq, read_sentence)


import json
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/api")
def api():
    seq, read = sentence_to_seq(str(request.args.get('s')))
    out = decode_sequence(seq)
    d = {
        'read_sentence': read,
        'out_sentence': out
    }
    return json.dumps(d)

@app.route("/web")
def web():
    sentence, _ = sentence_to_seq(str(request.args.get('s')))
    out = ""
    if sentence is not None and sentence is not "":
        out = decode_sequence(sentence)
        # print("output:", out)
    return render_template('app.html', output=out)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)