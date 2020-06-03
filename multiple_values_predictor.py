import numpy as np
import random
import sys
import io

from probability_sequence_learner import ProbabilitySequenceLearner

with io.open("simple-text.txt") as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

learner = ProbabilitySequenceLearner(char_indices, indices_char, 4)
learner.learn_sentences(text)

maxlen = 10
start_index = 0  # random.randint(0, len(text) - maxlen - 1)
primer_text = text[start_index: start_index + maxlen]

learner.generate_from_text(primer_text, 400)
