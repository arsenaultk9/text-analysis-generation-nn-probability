import numpy as np
import sys


class CombinedCellSequenceLearner:
    def __init__(self, text, memory_length):
        super().__init__()

        self.text = text

        chars = sorted(list(set(text)))
        print('total chars:', len(chars))

        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))
        self.memory_length = memory_length

        self.char_to_char_proba = np.zeros(
            shape=(len(self.char_indices) * len(self.char_indices), len(self.char_indices)))

    def learn_sentences(self):
        print("=== Counting references of letters to letters ===")

        char_to_char_counter = np.zeros(
            shape=(len(self.char_indices) * len(self.char_indices), len(self.char_indices)))

        for combined_char_index in range(len(self.text) - 2):
            cur_char = self.text[combined_char_index]
            next_char = self.text[combined_char_index + 1]
            pred_char = self.text[combined_char_index + 2]

            combined_char_position = self.char_indices[cur_char] * len(
                self.char_indices) + self.char_indices[next_char]

            cur_char_probabilities = char_to_char_counter[combined_char_position]
            cur_char_probabilities[self.char_indices[pred_char]] += 1

        print("=== Tranforming to probabilities the total char to chars ===")

        for combined_char_index, char_counter in enumerate(char_to_char_counter):
            total_usages = np.sum(char_counter)
            if total_usages == 0:
                continue

            for char_to_char_index, char_to_char_usage in enumerate(char_counter):
                self.char_to_char_proba[combined_char_index,
                                        char_to_char_index] = char_to_char_usage / total_usages

    def generate_from_text(self, primer_text, char_length):
        sentence = primer_text
        generated = sentence

        print('----- Generating with seed: "' + sentence + '"')

        sys.stdout.write(sentence)
        sys.stdout.flush()

        for i in range(char_length):
            combined_char_index = self.char_indices[sentence[-2]] * len(
                self.char_indices) + self.char_indices[sentence[-1]]

            next_char_probab = self.char_to_char_proba[combined_char_index]
            next_index = np.where(next_char_probab ==
                                  np.max(next_char_probab))[0][0]

            next_char = self.indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
