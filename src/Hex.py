import numpy as np
import random

from MonteCarloTreeSearch import MonteCarloTreeSearch
from action.HexMove import HexMove
from node.TwoPlayerMonteCarloTreeSearchNode import TwoPlayerMonteCarloTreeSearchNode
from state.HexGameState import HexGameState


class Hex:

    def __init__(self, board_representation, start_player, num_simulations, verbose=False):
        self.board = []
        self.verbose = verbose
        self.num_simulations = num_simulations
        self._create_board(board_representation)
        self.root = TwoPlayerMonteCarloTreeSearchNode(state=HexGameState(state=self.board, next_to_move=start_player, action_that_resulted_in_the_current_state=None), parent=None)

    def run(self):
        while not self.is_finished():
            mcts = MonteCarloTreeSearch(self.root)
            best_node = mcts.best_action(self.num_simulations)

            if self.verbose:
                best_node.print_move()

            self.root = best_node

    def move(self, action):
        new_state = self.root.state.move(action)
        self.root = TwoPlayerMonteCarloTreeSearchNode(state=new_state, parent=self.root)


    """         1000000000                 and      0110000000              """
    """                                                                     """
    """         is intrpreted as:                   as:                     """
    """                                                                     """
    """         Player 1's turn                     Player 2's turn         """
    """                                                                     """
    """            (00)                                 (10)                """
    """            /  \                                 /  \                """
    """         (00)--(00)                           (00)--(00)             """
    """            \  /                                 \  /                """
    """            (00)                                 (00)                """

    def _create_board(self, board_representation):

        size = int(np.sqrt((len(board_representation) - 2) / 2))

        bits = []

        for letter in board_representation:
            bits.append(int(letter))

        player = bits[1]  # Denne brukes ikke

        boardBits = bits[2:]

        board = [[[boardBits[(size * i + j) * 2], boardBits[(size * i + j) * 2 + 1]] for j in range(size)] for i in
                 range(size)]

        self.board = board

    def get_root(self):
        return self.root

    def is_finished(self):
        return self.root.state.is_game_over()

