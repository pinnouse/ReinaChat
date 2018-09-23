from __future__ import unicode_literals
from os import path
from io import open
import string
import collections
import re

import nltk

import torch

import csv
import configparser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

here = path.dirname(__file__)

config = configparser.ConfigParser()
config.read('config.ini')

class TextData():
  def __init__(self):
    # (key, value) => (word, (tag, frequency))
    self.input_words = dict()
    self.target_words = dict()

    max_samples = int(config['data']['samples'])
    vocab_size = int(config['data']['vocab_size'])
    data_path = config['data']['data_path']
    input_files = csv.reader([config['data']['in_data']])
    target_files = csv.reader([config['data']['out_data']])

    print("Reading input file(s)...")
    row_count = 0
    for fields in input_files:
      for i_file in fields:
        with open(data_path + i_file, 'r', encoding='utf-8', errors='ignore') as f:
          for line in f:
            if row_count < max_samples:
              for (w, tag) in self.parseSentence(line + " <EOS>"):
                if w in self.input_words:
                  self.input_words[w][1] += 1
                elif len(self.input_words) < vocab_size:
                  self.input_words[w] = [tag, 0]
              row_count += 1
            else:
              break
    print("Input rows read:", row_count)
    
    print("Reading target file(s)...")
    row_count = 0
    for fields in target_files:
      for t_file in fields:
        with open(data_path + t_file, 'r', encoding='utf-8', errors='ignore') as f:
          for line in f:
            if row_count < max_samples:
              for (w, tag) in self.parseSentence(line + " <EOS>"):
                if w in self.target_words:
                  self.target_words[w][1] += 1
                elif len(self.target_words) < vocab_size:
                  self.target_words[w] = [tag, 0]
              row_count += 1
            else:
              break
    print("Target rows read:", row_count)
    
    # Trim dictionaries so that the sizes are the same
    self.input_words = dict(list(self.input_words.items())[:min([vocab_size, len(self.input_words), len(self.target_words)])])
    self.target_words = dict(list(self.target_words.items())[:min([vocab_size, len(self.input_words), len(self.target_words)])])

    print("SORTING WORDS")
    # The word items are now dictionaries of type (key, value) => (word, index)
    self.input_words = self.sortWords(self.input_words)
    self.target_words = self.sortWords(self.target_words)
    print("WORDS SORTED")

  def sortWords(self, dict_of_words):
    '''Sorts a dictionary into a dictionary in the form of (key, value) => (word, index)
    from a dictionary that looks like (key, value) => (word, [word_tag, frequency])
    Args:
      dict_of_words: The dictionary as mentioned before
    Returns:
      Dict (key, value) => (word, index)
    '''

    # RegEx patterns to set custom tags that NLTK does not have
    extra_pattern = re.compile(r'<[A-Z]+>')
    commands_pattern = re.compile(r'\$[^ \d@#$%^&*()]{2,}')
    symbol_pattern = re.compile(r'[,;:@#$%^&*()\'\"]+')

    words_by_tag = dict()
    for w, (t, f) in dict_of_words.items():
      if len(extra_pattern.findall(w)):
        t = "A_EXTRA_SYMBOLS" # Such as <SOS>, <UNK>, <EOS>, <PRD> ...
      elif len(commands_pattern.findall(w)):
        t = "A_SPECIAL_COMMANDS"
      elif len(symbol_pattern.findall(t)):
        t = "SYMBL"

      if not t in words_by_tag.keys():
        words_by_tag[t] = []
      words_by_tag[t].append((w, f))

    words_by_tag['A_EXTRA_SYMBOLS'].append(('<SOS>', 0)) # Start of sequence add to 'extra' tag
    words_by_tag['A_EXTRA_SYMBOLS'].append(('<UNK>', 0)) # Unknown token add to 'extra' tag

    words_by_tag = collections.OrderedDict(sorted(words_by_tag.items()))

    # Now sort the words by frequency within their tags
    end_list = dict()
    i = 0
    for tag in words_by_tag:
      sorted_list = sorted(words_by_tag[tag], key=lambda x:x[1], reverse=True)
      for word, _ in sorted_list:
        end_list[word] = i
        i += 1

    return end_list

  def parseSentence(self, sentence):
    '''This function replaces special characters such as
    punctuation and special characters to be used by nltk for tagging
    Args:
      sentence: The sentence to be parsed
    Returns:
      List of words and tags in the form [(word, tag), (word, tag), ...]
    '''
    emoji_pattern = re.compile(r'([\U00010000-\U0010ffff])', flags=re.UNICODE)
    sentence = emoji_pattern.sub(r'\1 ', sentence)
    sentence = re.sub(r'\.', r' <PRD> ', sentence)
    sentence = re.sub(r'\!', r' <EXCL> ', sentence)
    sentence = re.sub(r'\?', r' <QSTN> ', sentence)
    sentence = re.sub(r',', r' <COMMA> ', sentence)
    sentence = re.sub(r'([;:])', r' \1', sentence)
    sentence = re.sub(r'([a-zA-Z]\'[a-zA-Z]*)', r'\1 ', sentence)
    return nltk.pos_tag(sentence.split())

  def tensorFromSentence(self, word_data, sentence):
    sentence += " <EOS>"
    tensor = torch.tensor([word_data[word] if word in word_data else word_data["<UNK>"] for word, _ in self.parseSentence(sentence)], dtype=torch.long, device=device).view(-1, 1)
    return tensor

  def sentenceFromTensor(self, word_data, tensor):
    words = []
    for i in tensor.tolist():
      for w, ind in word_data.items():
        if i[0] == ind:
          words.append(w)
    return ' '.join(words)



tData = TextData()
print(tData.sentenceFromTensor(tData.input_words, tData.tensorFromSentence(tData.input_words, "hello there.")))
print(len(tData.input_words))
print(len(tData.target_words))

run = True
while run:
  try:
    # hi
    i = 0
  except (SystemError, SystemExit, KeyboardInterrupt):
    run = False