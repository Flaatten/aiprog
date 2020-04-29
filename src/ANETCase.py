import numpy as np
import torch


class ANETCase:

    def __init__(self, x_input, target_distribution): # x_input = TwoPlayerMTCSNode
        self.x_input = x_input
        self.target_distribution = target_distribution

    def get_as_numpy_arrays(self):
        x = self.x_input.state.as_list()
        # player_to_move = self.x_input.state.next_to_move
        #
        # x = []
        #
        # if player_to_move == 0:  # [1, 0]
        #     x.append(1)
        #     x.append(0)
        # else:  # [0, 1]
        #     x.append(0)
        #     x.append(1)
        #
        # # two bits per cell to indicate player; [0,0] = empty, [1,0] = player1, [0,1] = player2
        # # board indexed by [row][col][player]
        # for row in self.x_input.state.board:
        #     for col in row:
        #         x.append(col[0])
        #         x.append(col[1])

        return x, self.target_distribution

    @staticmethod
    def process_state(state):
        x = state.as_list()

        x_train = np.array(x)

        return torch.from_numpy(x_train).float()