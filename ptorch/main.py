from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import os.path

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

here = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_TOKEN = 0 # Start of sequence token index
EOS_TOKEN = 1 # End of sequence token index
UNKNOWN_TOKEN = 2

# Language processing
class Data:
  def __init__(self, name):
    self.name = name
    self.word2count = {}
    self.index2word = {SOS_TOKEN: "SOS", EOS_TOKEN: "EOS", UNKNOWN_TOKEN: "UNK"}
    self.word2index = {w: ind for (w, ind) in self.index2word}
    self.n_words = len(self.index2word)

  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s      

def readData(i_file, t_file, samples=2000, data_path="data", reverse=False):
  print("Reading lines...")

  lines_in = []
  lines_target = []

  data_path = data_path + os.sep

  print("Reading input file")
  with open(os.path.join(here, data_path + i_file), encoding='utf-8', errors='ignore') as f:
    for n_row, row in enumerate(f):
      if n_row < samples:
        lines_in.append(row)
      else:
        break
  
  print("Reading target file")
  with open(os.path.join(here, data_path + t_file), encoding='utf-8', errors='ignore') as f:
    for n_row, row in enumerate(f):
      if n_row < samples:
        lines_target.append(row)
      else:
        break

  samp_size = min([len(lines_in), len(lines_target), samples])

  pairs = [[normalizeString(lines_in[i]), normalizeString(lines_target[i])] for i in range(samp_size)]

  if reverse:
    pairs = [list(reversed(p)) for p in pairs]
    i_data = Data('target')
    t_data = Data('in')
  else:
    i_data = Data('in')
    t_data = Data('target')

  return i_data, t_data, pairs

MAX_LENGTH = 20

def filterPair(p):
  return len(p[0].split(' ')) < MAX_LENGTH and \
    len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
  return [pair for pair in pairs if filterPair(pair)]

def prepareData(i_file, t_file, samples=2000, data_path="data", reverse=False):
  data_path = data_path + os.sep
  i_data, t_data, pairs = readData(i_file, t_file, samples, data_path, reverse)
  print("Read %d sentence pairs" % len(pairs))
  pairs = filterPairs(pairs)
  print("Trimmed to %d sentence pairs" % len(pairs))
  print("Counting words...")
  for pair in pairs:
    i_data.addSentence(pair[0])
    t_data.addSentence(pair[1])
  print("Counted words:")
  print(i_data.name, i_data.n_words)
  print(t_data.name, t_data.n_words)
  return i_data, t_data, pairs

i_data, t_data, pairs = prepareData("train.to", "train.from", reverse=True)

class EncoderRNN(nn.Module):
  def __init__(self, i_size, h_size):
    super(EncoderRNN, self).__init__()
    self.h_size = h_size # Hidden neurons

    self.embedding = nn.Embedding(i_size, h_size)
    self.gru = nn.GRU(h_size, h_size)

  def forward(self, input, hidden):
    embedded = self.embedding(input).view(1, 1, -1)
    output = embedded
    output, hidden = self.gru(output, hidden)
    return output, hidden
  
  def initHidden(self):
    return torch.zeros(1, 1, self.h_size, device=device)
  
class DecoderRNN(nn.Module):
  def __init__(self, h_size, o_size):
    super(DecoderRNN, self).__init__()
    self.h_size = h_size
    
    self.embedding =  nn.Embedding(o_size, h_size)
    self.gru = nn.GRU(h_size, h_size)
    self.out = nn.Linear(h_size, o_size)
    self.softmax = nn.LogSoftmax(dim=1)
  
  def forward(self, input, hidden):
    output = self.embedding(input).view(1, 1, -1)
    output = F.relu(output)
    output, hidden = self.gru(output, hidden)
    output = self.softmax(self.out(output[0]))
    return output, hidden

  def initHidden(self):
    return torch.zeros(1, 1, self.h_size, device=device)

class AttnDecoderRNN(nn.Module):
  def __init__(self, h_size, o_size, dropout_p=0.1, max_length=MAX_LENGTH):
    super(AttnDecoderRNN, self).__init__()
    self.h_size = h_size
    self.o_size = o_size
    self.dropout_p = dropout_p
    self.max_length = max_length

    self.embedding = nn.Embedding(self.o_size, self.h_size)
    self.attn =  nn.Linear(self.h_size * 2, self.max_length)
    self.attn_combine = nn.Linear(self.h_size * 2, self.h_size)
    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(self.h_size, self.h_size)
    self.out = nn.Linear(self.h_size, self.o_size)

  def forward(self, input, hidden, encoder_outputs):
    embedded = self.embedding(input).view(1, 1, -1)
    embedded = self.dropout(embedded)

    attn_weights = F.softmax(
      self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
    attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)

    output = F.relu(output)
    output, hidden = self.gru(output, hidden)

    output = F.log_softmax(self.out(output[0]), dim=1)
    return output, hidden, attn_weights

  def initHidden(self):
    return torch.zeros(1, 1, self.h_size, device=device)

def indexFromSentence(data, sentence):
  return [data.word2index[word] if word in data.word2index.keys() else data.word2index["UNK"] for word in sentence.split(' ')]

def tensorFromSentence(data, sentence):
  indeces = indexFromSentence(data, sentence)
  indeces.append(EOS_TOKEN)
  return torch.tensor(indeces, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
  i_tensor = tensorFromSentence(i_data, pair[0])
  t_tensor = tensorFromSentence(t_data, pair[1])
  return (i_tensor, t_tensor)

# Ratio to use teacher forcing (higher is more)
teacher_forcing_ratio = 0.5

def train(i_tensor, t_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
  encoder_hidden = encoder.initHidden()

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  input_length = i_tensor.size(0)
  target_length = t_tensor.size(0)

  encoder_outputs = torch.zeros(max_length, encoder.h_size, device=device)

  loss = 0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(
      i_tensor[ei], encoder_hidden
    )
    encoder_outputs[ei] = encoder_output[0, 0]

  decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

  decoder_hidden = encoder_hidden

  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  if use_teacher_forcing:
    # Teacher forcing; feed target as new input
    for di in range(target_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_outputs
      )
      loss += criterion(decoder_output, t_tensor[di])
      decoder_input = t_tensor[di] #Teacher forcing

  else:
    for di in range(target_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_outputs
      )
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach() # Detach from history as input

      loss += criterion(decoder_output, t_tensor[di])
      if decoder_input.item() == EOS_TOKEN:
        break

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.item() / target_length


# Helpers for determining remaining time

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def saveCheckpoint(state, is_best, iteration, output_dir='output', filename='checkpoint'):
  """Save checkpoint if new best is achieved"""

  output_dir = output_dir + os.sep
  if is_best:
    print("=> Saving new best")
    torch.save(state, os.path.join(here, output_dir + filename + '-' + iteration + '.pth.tar')) # Save checkpoint state
  else:
    print("=> Validation accuracy did not improve")

def trainIters(encoder, decoder, n_iters, start_iter=1, print_every=1000, ckpt_every=5000, learning_rate=1e-2):
  start = time.time()
  print_loss_total = 0 # Reset every print_every

  best_loss = None
  prev_best = None

  encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
  training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
  criterion = nn.NLLLoss()

  for iter in range(start_iter, n_iters):
    training_pair = training_pairs[iter - 1]
    i_tensor = training_pair[0]
    t_tensor = training_pair[1]

    loss = train(i_tensor, t_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    print_loss_total += loss


    if best_loss == None or loss < best_loss:
      best_loss = loss
      best_states = {
        'iter': start_iter + iter + 1,
        'enc_state_dict': encoder.state_dict(),
        'dec_state_dict': decoder.state_dict(),
        'best_loss': best_loss
      }

    if iter % ckpt_every == 0:
      is_best = bool(best_loss < prev_best) if prev_best != None else True
      saveCheckpoint(best_states, is_best, iter)
      prev_best = best_loss

    if iter % print_every == 0:
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0
      print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

def sampleRandomly(target_tensor, probability=0.85, threshold=0.92):
  top_scores = target_tensor.tolist()[0]
  index2scores = list(zip(range(len(top_scores)), top_scores)) # Zip by (word index, top score)
  index2scores = sorted(index2scores, key=lambda x: x[1], reverse=True) # Sort by scores

  selected_index, selected_score = index2scores[0]
  for i, score in index2scores:
    if i != selected_index:
      score_proportion = score / selected_score
      if score_proportion < threshold and random.random() < probability * abs(score_proportion): # If above threshold and probability (weighted towards score)
        selected_index = i
  
  return selected_index

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH, use_random=False):
  with torch.no_grad():
    i_tensor = tensorFromSentence(i_data, sentence)
    input_length = i_tensor.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = torch.zeros(max_length, encoder.h_size, device=device)

    for ei in range(input_length):
      encoder_output, encoder_hidden = encoder(
        i_tensor[ei], encoder_hidden
      )
      encoder_outputs[ei] += encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

    decoder_hidden = encoder_hidden

    decoded_words = []

    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_outputs
      )
      decoder_attentions[di] = decoder_attention.data

      tempW = ""
      if use_random:
        randomSample = sampleRandomly(decoder_output)
        tempW = randomSample
        decoder_input = torch.tensor(tempW, device=device)
      else:
        topv, topi = decoder_output.data.topk(1)
        tempW = topi.item()
        decoder_input = topi.squeeze().detach()

      if tempW == EOS_TOKEN:
        decoded_words.append('<EOS>')
        break
      else:
        decoded_words.append(t_data.index2word[tempW])

    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
  for i in range(n):
    pair = random.choice(pairs)
    print('>', pair[0])
    print('=', pair[1])
    output_words, attentions = evaluate(encoder, decoder, pair[0], use_random=True)
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')

h_size = 256
encoder1 = EncoderRNN(i_data.n_words, h_size).to(device)
attn_decoder1 = AttnDecoderRNN(h_size, t_data.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75, print_every=5000)
evaluateRandomly(encoder1, attn_decoder1)

run = True
while run:
  try:
    output_words, attentions = evaluate(
      encoder1, attn_decoder1,
      input('>'),
      use_random=True
    )
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')
  except (KeyboardInterrupt, SystemExit):
    print ("Thanks for using this program")
    run = False