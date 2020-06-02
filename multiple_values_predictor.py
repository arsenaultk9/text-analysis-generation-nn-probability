import numpy as np
import random
import sys
import io

from probability_sequence_learner import ProbabilitySequenceLearner

with io.open("nostradamus-quatrains.txt") as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

learner = ProbabilitySequenceLearner(char_indices, indices_char, 3)
learner.learn_sentences(sentences)

start_index = random.randint(0, len(text) - maxlen - 1)
primer_text = text[start_index: start_index + maxlen]

learner.generate_from_text(primer_text, 400)
