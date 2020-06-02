import numpy as np
import sys


class ProbabilitySequenceLearner:
    def __init__(self, char_indices, indices_char, memory_length):
        super().__init__()

        self.char_indices = char_indices
        self.indices_char = indices_char
        self.memory_length = memory_length

        self.char_to_char_proba = np.zeros(
            shape=(memory_length, len(char_indices), len(char_indices)))

    def learn_sentences(self, sentences):
        print("=== Counting references of letters to letters ===")

        char_to_char_counter = np.zeros(
            shape=(self.memory_length, len(self.char_indices), len(self.char_indices)))

        for sentence in sentences:
            for char_index in range(len(sentence) - self.memory_length):
                cur_char = sentence[char_index]

                for current_mem_sequence in range(self.memory_length):
                    next_char = sentence[char_index + current_mem_sequence + 1]

                    cur_char_probabilities = char_to_char_counter[
                        current_mem_sequence][self.char_indices[cur_char]]
                    cur_char_probabilities[self.char_indices[next_char]] += 1

        print("=== Tranforming to probabilities the total char to chars ===")

        for sequence_index in range(self.memory_length):
            for char_index, char_counter in enumerate(char_to_char_counter[sequence_index]):
                total_usages = np.sum(char_counter)
                if total_usages == 0:
                    continue

                for char_to_char_index, char_to_char_usage in enumerate(char_counter):
                    self.char_to_char_proba[sequence_index,
                                            char_index,
                                            char_to_char_index] = char_to_char_usage / total_usages

    def generate_from_text(self, primer_text, char_length):
        sentence = primer_text
        generated = sentence

        print('----- Generating with seed: "' + sentence + '"')

        sys.stdout.write(sentence)
        sys.stdout.flush()

        for i in range(char_length):
            next_char_probab = np.zeros(shape=(len(self.char_indices)))

            # Cummulate backwards probability summations for next char.
            for current_mem_sequence in range(self.memory_length):
                current_index = -(current_mem_sequence + 1)
                cur_char_index = self.char_indices[sentence[current_index]]
                next_char_probab += self.char_to_char_proba[current_mem_sequence][cur_char_index]

            next_index = np.where(next_char_probab ==
                                  np.max(next_char_probab))[0][0]

            next_char = self.indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
