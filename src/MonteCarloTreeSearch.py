class MonteCarloTreeSearch:

    def __init__(self, root):
        self.root = root

    def best_action(self, num_simulations_to_run):
        for i in range(num_simulations_to_run):
            rollout_node = self.do_tree_policy()
            reward = rollout_node.rollout()
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