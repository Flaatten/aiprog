import numpy as np

from utils import generate_random_int_in_range, increment


class TwoPlayerMonteCarloTreeSearchNode:

    def __init__(self, state, parent):
        self.num_visits = 0
        self.state = state
        self.parent = parent
        self.children = []
        self.untried_actions = []
        self.results = {}

    def untried_actions(self):
        if len(self.untried_actions) == 0:
            self.untried_actions = self.state.get_legal_actions()
        return self.untried_actions

    def get_standardised_distribution(self):
        dist = []

        board_rep = self.state.get_board()

        for cell in board_rep.split():
            if cell is empty:  # TODO FIX WHEN MOAN HAS DECIDED UPON BOARD REPRESENTATION FORMAT
                for child in self.children:
                    if child.state.board[cell_index] == "VAL":  # child corresponds to current cell TODO FIX AFTER MOAN FINISHED ( FILL WITH FILL-VALUES )
                        wins = 0
                        win_key = child.parent.get_next_to_move()

                        if win_key in child.results:
                            wins = child.results.get(win_key)

                        dist.append(wins / child.n())
                        break  # continue with next cell

                dist.append(0) # TODO cell is empty, but not found by rollout. Perhaps not an ideal way to handle it?
            else:
                dist.append(0)  # cell is busy, probability for taking action = 0

        norm = [float(i) / sum(dist) for i in dist]

        return norm

    def n(self):
        return self.num_visits

    def q(self):
        wins = 0
        losses = 0

        win_key = self.state.parent.get_next_to_move()
        loss_key = -1 * self.state.parent.get_next_to_move()

        if win_key in self.results:
            wins = self.results.get(win_key)

        if loss_key in self.results:
            losses = self.results.get(loss_key)

        return wins - losses

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = TwoPlayerMonteCarloTreeSearchNode(next_state, self)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def backpropagate(self, reward):
        self.num_visits += 1
        increment(self.results, reward)

        if self.parent is not None:
            self.parent.backpropagate(reward)

    def rollout(self):  # TODO MAKE THIS USE THE NEURAL NET
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)

        return current_rollout_state.game_result()

    def rollout_policy(self, possible_moves):  # TODO CHANGE TO NEURAL NET
        random_element_index = generate_random_int_in_range(0, len(possible_moves))
        return possible_moves.get(random_element_index)

    def get_state(self):
        return self.state

    def print_move(self):
        self.state.print_move()

    def is_fully_expanded(self):
        return len(self.untried_actions()) == 0

    def get_best_child(self, c_param):
        current_best_child = None
        current_best_score = -1 * float("inf")

        for child in self.children:

            score = (child.q() / child.n()) + c_param * np.sqrt((2 * np.log(self.n()) / child.n()))

            if score > current_best_score:
                current_best_child = child
                current_best_score = score

        return current_best_child


