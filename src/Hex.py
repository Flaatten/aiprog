from node.TwoPlayerMonteCarloTreeSearchNode import TwoPlayerMonteCarloTreeSearchNode
from state.HexGameState import HexGameState


class Hex():

    def __init__(self, board_representation, start_player, num_simulations, verbose=False):
        self.board = []
        self.verbose = verbose
        self.num_simulations = num_simulations
        self._create_board(board_representation)

        self.state = HexGameState(state=self.board, next_to_move=start_player, action_that_resulted_in_the_current_state=None)
        self.root = TwoPlayerMonteCarloTreeSearchNode(state=self.state, parent=None)


    def run(self):
        pass

    def _create_board(self, board_representation):
        pass # TODO

    def get_state(self):
        return self.state

    def get_root(self):
        return self.root
