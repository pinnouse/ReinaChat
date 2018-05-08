from __future__ import print_function
from io import open
import os
import os.path
import configparser
import operator
from collections import OrderedDict
import nltk
from nltk.tokenize import RegexpTokenizer
import vocab_builder
import random
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense

### CONSTANTS
here = os.path.dirname(__file__)

tokenizer = RegexpTokenizer(r'\w+|\<[A-Z]+\>|\$[a-z]+|&[a-z]+;|[a-z]?\'[a-z]+|[.?!]') # Special commands are: $command
blacklist_pattern = r'http://[a-z]*'

config = configparser.ConfigParser()
conf_file = os.path.join(here, 'bot-config.ini')
config.read(conf_file)

batch_size = int(config['DEFAULT']['batch_size'])
epochs = int(config['DEFAULT']['epochs'])
latent_dim = int(config['DEFAULT']['latent_dim'])
num_samples = int(config['DEFAULT']['num_samples'])

data_path = config['DEFAULT']['data_path']
vocab_size = int(config['DEFAULT']['vocab_size'])

max_seq_len = int(config['DEFAULT']['max_seq_len'])
input_texts = []
target_texts = []
input_words = dict([("<UNK>", 0)])
target_words = dict([("<GO>", 0), ("<UNK>", 0), ("<EOS>", 0)])
with open(os.path.join(here, data_path + 'custom.enc'), 'r', encoding='utf-8', errors='ignore') as f:
    line_enc = f.read().split('\n')
with open(os.path.join(here, data_path + 'custom.dec'), 'r', encoding='utf-8', errors='ignore') as f:
    line_dec = f.read().split('\n')
with open(os.path.join(here, data_path + 'train.from'), 'r', encoding='utf-8', errors='ignore') as f:
    for row in f:
        row.replace(blacklist_pattern, '')
        line_enc.append(row)
        if len(line_enc) >= num_samples:
            break
with open(os.path.join(here, data_path + 'train.to'), 'r', encoding='utf-8', errors='ignore') as f:
    for row in f:
        row.replace(blacklist_pattern, '')
        line_dec.append(row)
        if len(line_dec) >= num_samples:
            break

small_samp_size = min([num_samples, len(line_enc)-1, len(line_dec)-1])
if len(line_enc) > num_samples or len(line_dec) > num_samples:
    line_enc = line_enc[:small_samp_size]
    line_dec = line_dec[:small_samp_size]

for i in range(small_samp_size):
    input_text = line_enc[i].lower()
    target_text = "<GO> " + line_dec[i] + " <EOS>"
    
    # print(tokenizer.tokenize(input_text))
    if len(tokenizer.tokenize(input_text)) > max_seq_len:
        input_text = " ".join(tokenizer.tokenize(input_text)[:max_seq_len])
    if len(tokenizer.tokenize(target_text)) > max_seq_len:
        target_text = " ".join(tokenizer.tokenize(target_text)[:max_seq_len-1]) + " <EOS"

    for w in tokenizer.tokenize(input_text):
        if w not in input_words:
            if len(input_words) < vocab_size:
                input_words.update({w: 1})
            else:
                input_text.replace(w, "<UNK>")
                w="<UNK>"
        else:
            input_words[w] += 1
    for w in tokenizer.tokenize(target_text):
        if w not in target_words:
            if len(target_words) < vocab_size:
                target_words.update({w: 1})
            else:
                target_text.replace(w, "<UNK>")
                w="<UNK>"
        else:
            target_words[w] += 1

    input_texts.append(input_text)
    target_texts.append(target_text)

input_words = vocab_builder.build_vocab(input_words)
target_words = vocab_builder.build_vocab(target_words)
num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)

max_encoder_seq_length = max([len(tokenizer.tokenize(txt)) for txt in input_texts])
max_decoder_seq_length = max([len(tokenizer.tokenize(txt)) for txt in target_texts])

print("Samples:", min(len(input_text), len(target_text)))
print("Unique input tokens:", num_encoder_tokens)
#print("Input dictionary:", input_words)
print("Unique output tokens:", num_decoder_tokens)
#print("Target dictionary:", target_words)
print("Max seq length for input:", max_encoder_seq_length)
print("Max seq length for output:", max_decoder_seq_length)

# Dictionaries containing word to id
input_token_index = dict([w, i] for i, w in enumerate(input_words))
target_token_index = dict([w, i] for i, w in enumerate(target_words))

# Free memory
input_words = None
target_words = None
line_enc = None
line_dec = None


# Create three dimensional arrays
# For each sentence -> maximum words -> binary of word index in B.O.W on or off
encoder_input_data = np.zeros(
    shape=(len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32'
)
decoder_input_data = np.zeros(
    shape=(len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32'
)
decoder_target_data = np.zeros(
    shape=(len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32'
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, w in enumerate(tokenizer.tokenize(input_text)):
        if w not in input_token_index:
            w = "<UNK>"
        encoder_input_data[i, t, input_token_index[w]] = 1.
    for t, w in enumerate(tokenizer.tokenize(target_text)):
        if w not in target_token_index:
            w = "<UNK>"
        decoder_input_data[i, t, target_token_index[w]] = 1.
        if t > 0:
            decoder_target_data[i, t-1, target_token_index[w]] = 1.


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
if os.path.isfile(model_location):
    model.load_weights(model_location)
else:
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    model.save(model_location)

encoder_model = Model(encoder_inputs, encoder_states)

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

reverse_input_w_index = dict((i, w) for w, i in input_token_index.items())
reverse_target_w_index = dict((i, w) for w, i in target_token_index.items())

# def sample(a, temperature=1.0):
#     a = np.array(a)**(1/temperature)
#     p_sum = sum(a)
#     for i in range(len(a)):
#         a[i] = a[i]/p_sum
#     print('a:', sum(a))
#     return np.argmax(np.random.multinomial(1, a, 1))

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
        sampled_w = reverse_target_w_index[sampled_token_index]
        # print("sampled token index:", sampled_token_index, "word:", sampled_w)
        decoded_sentence += sampled_w + " "

        if sampled_w == "<EOS>" or len(decoded_sentence.split(' ')) > max_decoder_seq_length:
            stop_condition = True
        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]
    
    return decoded_sentence

def sentence_to_seq(sentence):
    sentence = tokenizer.tokenize(sentence)
    seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    
    read_sentence = ""
    for i in range(min(max_encoder_seq_length, len(sentence))):
        w = sentence[i].lower()
        if w not in list(reverse_input_w_index.values()):
            w = "<UNK>"
        seq[0, i, input_token_index[w]] = 1.
        read_sentence += w + " "
        
    # print("Read sentence: ", read_sentence)

    return (seq, read_sentence)

for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index:seq_index+1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

# b = Bot()
# while True:
#    try:
#        input_seq, _ = b.sentence_to_seq(input('You >> '))
#        decoded_sentence = b.decode_sequence(input_seq)
#        print('Bot >>', decoded_sentence)
#    except (KeyboardInterrupt, SystemExit):
#        raise
