import numpy as np
import random
import sys
import io

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


print("=== Counting references of letters to letters ===")
char_to_char_counter = np.zeros(shape=(len(char_indices), len(char_indices)))

for sentence in sentences:
    for char_index in range(len(sentence) - 1):
        cur_char = sentence[char_index]
        next_char = sentence[char_index + 1]

        cur_char_probabilities = char_to_char_counter[char_indices[cur_char]]
        cur_char_probabilities[char_indices[next_char]] += 1

print("=== Tranforming to probabilities the total char to chars ===")

char_to_char_proba = np.zeros(shape=(len(char_indices), len(char_indices)))
for char_index, char_counter in enumerate(char_to_char_counter):
    total_usages = np.sum(char_counter)

    for char_to_char_index, char_to_char_usage in enumerate(char_counter):
        char_to_char_proba[char_index,
                           char_to_char_index] = char_to_char_usage / total_usages

print("=== Generating sentence from existing sentence ===")

start_index = random.randint(0, len(text) - maxlen - 1)
sentence = text[start_index: start_index + maxlen]
generated = sentence

print('----- Generating with seed: "' + sentence + '"')

for i in range(400):
    cur_char_index = char_indices[sentence[-1]]
    next_char_probab = char_to_char_proba[cur_char_index]
    next_index = np.where(next_char_probab == np.max(next_char_probab))[0][0]

    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()
print()
