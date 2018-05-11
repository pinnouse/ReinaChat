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

SOS_TOKEN = 0 # Start of sequence token index
EOS_TOKEN = 1 # End of sequence token index

# Language processing
class Data:
  def __init__(self, name):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "SOS", 1: "EOS"}
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

def readData(i_file, t_file, samples=2000, data_path="data/", reverse=False):
  print("Reading lines ------")

  lines_in = []
  lines_target = []
  with open(os.path.join(here, data_path + i_file), encoding='utf-8', errors='ignore') as f:
    for n_row, row in enumerate(f):
      if n_row < samples:
        lines_in[n_row] = row
      else:
        break
  
  with open(os.path.join(here, data_path + t_file), encoding='utf-8', errors='ignore') as f:
    for n_row, row in enumerate(f):
      if n_row < samples:
        lines_target[n_row] = row
      else:
        break

  samp_size = min([len(lines_in), len(lines_target), samples])

  pairs = [[lines_in[i], lines_target[i]] for i in range(samp_size)]

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

def prepareData(i_file, t_file, samples=2000, data_path="data/", reverse=False):
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

i_data, t_data, pars = prepareData("train.to", "train.from", True)