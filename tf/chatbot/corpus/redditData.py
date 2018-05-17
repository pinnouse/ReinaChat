import os

class RedditData:

  def __init__(self, commentsFile):
    self.conversations = []
    self.loadLines(commentsFile)

  def loadLines(self, fileName):
    in_lines = []
    out_lines = []
    with open(fileName + ".from", 'r', encoding='utf-8', errors='ignore') as f:
      for line in f:
        in_lines.append(line)
    with open(fileName + ".to", 'r', encoding='utf-8', errors='ignore') as f:
      for line in f:
        out_lines.append(line)

    shortest = min(len(in_lines), len(out_lines)) - 1
    in_lines = in_lines[:shortest]
    out_lines = out_lines[:shortest]
    for i in range(shortest):
      self.conversations.append({
        "lines": [
          { "text": in_lines[i] },
          { "text": out_lines[i] }
        ]
      })

  def getConversations(self):
    return self.conversations