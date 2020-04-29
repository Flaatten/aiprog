import numpy as np
import random

from ANETCase import ANETCase
from action.HexMove import HexMove


class MonteCarloTreeSearch:

    def __init__(self, root, neural_net, greedy_prob):
        self.root = root
        self.neural_net = neural_net
        self.greedy_prob = greedy_prob # epsilon

    def best_action(self, num_simulations_to_run):
        for i in range(num_simulations_to_run):
            rollout_node = self.do_tree_policy()
            reward = self.rollout()
            rollout_node.backpropagate(reward)
        return self.root.get_best_child(1.0)

    def do_tree_policy(self):
        current_node = self.root

        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.get_best_child(1.0)

        return current_node

    def rollout(self):
        current_rollout_state = self.root.state
        while not current_rollout_state.is_game_over():
            neural_net_input = ANETCase.process_state(current_rollout_state)
            y_predicted = self.neural_net.forward(neural_net_input).tolist()
            y_predicted = current_rollout_state.set_invalid_actions_to_zero_from_list(y_predicted)
            norm = [float(i) / sum(y_predicted) for i in y_predicted]

            if random.random() <= self.greedy_prob: # random move
                rand_num = random.random()
                action = self.root.state.get_random_weighted_move(norm)
            else: # greedy move
                action = self.root.state.get_greedy_move(norm)

            current_rollout_state = current_rollout_state.move(action)

        return current_rollout_state.game_result()

