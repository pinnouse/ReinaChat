# Used to clean up the training information
# Usage: python data_parser.py <input training> <target training> <name of encoded output>
import os.path as op
import sys

i = []
o = []
with open(op.join(op.dirname(__file__), sys.argv[1]), 'r') as f:
    i = f.read().split('\n')
with open(op.join(op.dirname(__file__), sys.argv[2]), 'r') as f:
    o = f.read().split('\n')
with open(op.join(op.dirname(__file__), sys.argv[3]), 'w') as f:
    for ind in range(min(len(i), len(o))):
        if i[ind] != '' and o[ind] != '':
            f.write(i[ind] + '+++$+++' + o[ind] + '\n')
