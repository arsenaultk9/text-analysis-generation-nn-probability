import numpy as np
import sys


class CombinedCellSequenceLearner:
    def __init__(self, text, memory_length, cell_size):
        super().__init__()

        self.text = text

        chars = sorted(list(set(text)))
        print('total chars:', len(chars))

        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))

        self.memory_length = memory_length
        self.cell_size = cell_size

        self.char_to_char_proba = np.zeros(
            shape=(memory_length, len(self.char_indices) ** cell_size, len(self.char_indices)))

    def learn_sentences(self):
        print("=== Counting references of letters to letters ===")

        char_to_char_counter = np.zeros(
            shape=(self.memory_length, len(self.char_indices) ** self.cell_size, len(self.char_indices)))

        memory_buffer = self.memory_length - 1

        for combined_char_index in range(len(self.text) - self.cell_size - memory_buffer):
            pred_char = self.text[combined_char_index +
                                  self.cell_size + memory_buffer]

            for current_mem_sequence in range(self.memory_length):
                cell_chars = []

                for cell_char_pos in range(self.cell_size):
                    cell_chars.append(
                        self.text[combined_char_index + current_mem_sequence + cell_char_pos])

                combined_char_position = self.get_combined_char_pos(cell_chars)

                cur_char_probabilities = char_to_char_counter[current_mem_sequence][combined_char_position]
                cur_char_probabilities[self.char_indices[pred_char]] += 1

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
                memory_pointer = self.memory_length - current_mem_sequence - 1
                current_index = -(current_mem_sequence + 1)

                cell_chars = []

                for cell_char_pos in range(self.cell_size):
                    cell_position_in_past = - \
                        (self.cell_size - cell_char_pos - 1)
                    cell_chars.append(
                        sentence[current_index + cell_position_in_past])

                combined_char_position = self.get_combined_char_pos(cell_chars)

                char_proba_time_ajusted = self.char_to_char_proba[memory_pointer][combined_char_position] * \
                    (1 / (self.memory_length - memory_pointer))

                next_char_probab += char_proba_time_ajusted

            next_index = np.where(next_char_probab ==
                                  np.max(next_char_probab))[0][0]

            next_char = self.indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

    def get_combined_char_pos(self, chars):
        combined_char_position = 0

        for char_index, cur_char in enumerate(chars):
            position_power = self.cell_size - char_index - 1
            combined_char_position += self.char_indices[cur_char] * (len(
                self.char_indices) ** position_power)

        return combined_char_position
