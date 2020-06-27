import numpy as np
import random
import sys
import io

from single_cell_sequence_learner import SingleCellSequenceLearner
from combined_cell_sequence_learner import CombinedCellSequenceLearner

with io.open("more-complex-text.txt") as f:
    text = f.read().lower()
print('corpus length:', len(text))

learner = CombinedCellSequenceLearner(text, 2)
learner.learn_sentences()

maxlen = 10
start_index = 0  # random.randint(0, len(text) - maxlen - 1)
primer_text = text[start_index: start_index + maxlen]

learner.generate_from_text(primer_text, 400)
