from MonteCarloTreeSearch import MonteCarloTreeSearch
from node.TwoPlayerMonteCarloTreeSearchNode import TwoPlayerMonteCarloTreeSearchNode
from state.HexGameState import HexGameState


class Hex:

    def __init__(self, board_representation, start_player, num_simulations, verbose=False):
        self.board = []
        self.verbose = verbose
        self.num_simulations = num_simulations
        self._create_board(board_representation)

        self.root = TwoPlayerMonteCarloTreeSearchNode(state=HexGameState(state=self.board, next_to_move=start_player, action_that_resulted_in_the_current_state=None), parent=None)

    # Used for simulations in fictitious game
    def run(self):
        while not self.is_finished():
            mcts = MonteCarloTreeSearch(self.root)
            best_node = mcts.best_action(self.num_simulations)

            if self.verbose:
                best_node.print_move()

            self.root = best_node

    # Used for moves in the actual game
    def move(self, action):
        new_state = self.root.state.move(action)
        self.root = TwoPlayerMonteCarloTreeSearchNode(state=new_state, parent=self.root)

    def _create_board(self, board_representation):
        pass  # TODO

    def get_root(self):
        return self.root

    def is_finished(self):
        return self.root.state.is_game_over()
