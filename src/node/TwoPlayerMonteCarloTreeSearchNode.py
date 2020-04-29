import random

import numpy as np

from action.HexMove import HexMove
from utils import generate_random_int_in_range, increment


class TwoPlayerMonteCarloTreeSearchNode:

    def __init__(self, state, parent):
        self.num_visits = 0
        self.state = state
        self.parent = parent
        self.children = []
        self.untried_actions = None
        self.results = {0: 0,  # 0 = losses, 1 = wins
                        1: 0
                        }

    def _untried_actions(self):
        if self.untried_actions is None:
            self.untried_actions = self.state.get_legal_actions()
        return self.untried_actions

    def get_standardised_distribution(self):
        dist = [0] * (len(self.state.board) ** 2)

        for row_i in range(len(self.state.board)):
            for col_j in range(len(self.state.board)):
                if self.state.is_open_cell([row_i, col_j]):
                    for child in self.children:
                        if not child.state.is_open_cell([row_i, col_j]):
                            dist_index = 3 * row_i - 1 + (col_j + 1)

                            dist[dist_index] = child.n()

        norm = [float(i) / sum(dist) if i != 0 else i for i in dist]

        return norm

    def n(self):
        return self.num_visits

    def q(self):
        wins = 0
        losses = 0

        win_key = self.parent.state.next_to_move
        loss_key = 1 - self.parent.state.next_to_move

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

    def backpropagate(self, reward):  # next_to_move = 0 if player 1 to move, and 1 if player 2 to move
        self.num_visits += 1

        if isinstance(reward, list):
            if (reward[0] == 1) and (self.state.next_to_move == 1):  # i won!
                reward = 1
            else:
                reward = 0

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
        return possible_moves[random_element_index]

    def get_state(self):
        return self.state

    def is_fully_expanded(self):
        return len(self._untried_actions()) == 0

    def get_best_child(self, c_param):
        current_best_child = None
        current_best_score = -1 * float("inf")

        for child in self.children:

            score = (child.q() / child.n()) + c_param * np.sqrt((2 * np.log(self.n()) / child.n()))

            if score > current_best_score:
                current_best_child = child
                current_best_score = score

        return current_best_child

