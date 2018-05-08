import nltk
import re
from collections import OrderedDict
import operator

special_tokens = [
    r"<[A-Z]+>",
    r"\$[a-z]+"
]

special_words = [
    "lol",
    "lel",
    "lul",
    "lmao",
    "xd"
]

def build_vocab(dict_words):
    # words will come in pairs of (word, frequency)
    tagged = nltk.pos_tag(list(dict_words.keys()))
    words_tagged = {}
    for w, tag in tagged:
        for regex in special_tokens:
            if re.match(regex, w): tag = ".SPCL"
        
        # Special exceptions
        if w.lower() in special_words:
            tag = "SLANG"

        if tag in list(words_tagged.keys()):
            words_tagged[tag] += " " + w
        else:
            words_tagged[tag] = w
    vocab = []
    for tag in sorted(list(words_tagged.keys())):
        w = words_tagged[tag].split(" ")
        tagged_word_in_dict = {x: freq for x, freq in dict_words.items() if x in w}
        
        vocab += (OrderedDict(sorted(tagged_word_in_dict.items(), key=operator.itemgetter(1), reverse=True)).keys())
    # print(words_tagged)
    return vocab