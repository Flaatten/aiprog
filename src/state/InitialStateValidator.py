import numpy as np

class InitialStateValidator:

    @staticmethod
    def is_valid(state):

        size = len(state)

        for i in range(size):
            if len(state[i]) != size:
                return False
            for j in range(size):
                if state[i][j][0] + state[i][j][1] > 1:
                    return False
                for w in range(2):
                    if state[i][j][w] != 0 and state[i][j][w] != 1:
                        return False
        return True