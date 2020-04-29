import numpy as np

class InitialStateValidator:

    @staticmethod
    def is_valid(state):

        # State is the string representing the board

        bits = []
        for letter in state:
            bits.append(int(letter))

        if np.sqrt(len(bits)/2 - 1) % 1 == 0:
            for i in range(len(bits)/2):
                if bits(i) == 0 or bits(i) == 1 and bits(i + 1) == 0 or bits(i + 1) == 1 and bits(i) + bits(i + 1) <= 1:
                    return True

        return False
