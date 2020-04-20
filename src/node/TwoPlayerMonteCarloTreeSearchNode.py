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

        self.children.add(child_node)
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

    def get_state(self):
        return self.state

    def print_move(self):
        self.state.print_move()

    def untried_actions(self):
        pass  # TODO

    def n(self):
        pass

    def q(self):
        pass

    def expand(self):
        pass

    def is_terminal_node(self):
        pass

    def backpropagate(self, reward):
        pass

    def rollout(self):
        pass

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

    def rollout_policy(self, possible_moves):
        random_element_index = generate_random_int_in_range(0, len(possible_moves))
        return possible_moves.get(random_element_index)
