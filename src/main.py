from ANETCase import ANETCase
from ANETDataset import ANETDataset
from ActorNet import ActorNet
from Hex import Hex
from MonteCarloTreeSearch import MonteCarloTreeSearch
from ReplayBuffer import ReplayBuffer
from constant.ANETOptimizer import ANETOptimizer
from constant.ActivationFunction import ActivationFunction
from state.InitialStateValidator import InitialStateValidator

import copy


def test():
    size = 3
    number_of_episodes = 10
    number_of_search_games_per_actual_move = 100
    learning_rate = 0.01
    hidden_layers_structure = [5, 5, 5]
    hidden_nodes_activation_function = ActivationFunction.RELU
    ANET_optimizer = ANETOptimizer.SGD
    number_of_ANETs_to_cache_for_TOPP = 1
    num_games_to_be_played_between_two_agenst_during_TOPP = 2
    save_interval_for_ANET = 40
    number_of_cases_from_buffer_for_intertraining = 10

    buffer = ReplayBuffer()  # TODO INPUT DIMS TO NEURAL NET NOT CORRECT
    net = ActorNet(input_dim=size ** 2, hidden_layers=hidden_layers_structure, output_dim=size ** 2, learning_rate=learning_rate, optimizer=ANET_optimizer, hidden_nodes_activation_function=hidden_nodes_activation_function)

    for n in range(number_of_episodes):
        game = Hex("0" * (size ** 2), 1, number_of_search_games_per_actual_move)
        if InitialStateValidator.is_valid(game.root.state):

            while not game.is_finished():
                s_init = copy.deepcopy(game.root)
                mcts = MonteCarloTreeSearch(s_init)

                for i in range(number_of_search_games_per_actual_move):
                    rollout_node = mcts.do_tree_policy()
                    reward = rollout_node.rollout()
                    rollout_node.backpropagate(reward)

                distribution = mcts.root.get_standardised_distribution()
                buffer.add_case(ANETCase(mcts.root, distribution))
                move = None # TODO
                game.move(move)

            train_data = ANETDataset(buffer.get_mini_batch(number_of_cases_from_buffer_for_intertraining))
            net.train_using_buffer_subset(train_data)

            if n % save_interval_for_ANET == 0:
                net.save_params(n)


        """
        (a) Initialize the actual game board (Ba) to an empty board.
        (b) s_init ← starting board state
        (c) Initialize the Monte Carlo Tree (MCT) to a single root, which represents s_init 
        (d) While Ba not in a final state:
            • Initialize Monte Carlo game board (Bmc) to same state as root. 
            • For gs in number search games:
                – Use tree policy Pt to search from root to a leaf (L) of MCT. Update Bmc with each move.
                # TODO – Use ANET to choose rollout actions from L to a final state (F). Update Bmc with each move. – Perform MCTS backpropagation from F to root.
            • next gs
            • D = distribution of visit counts in MCT along all arcs emanating from root. • Add case (root, D) to RBUF
            • Choose actual move (a*) based on D
            • Perform a* on root to produce successor state s*
            • Update Ba to s*
            • In MCT, retain subtree rooted at s*; discard everything else.
            • root←s*
        (e) Train ANET on a random minibatch of cases from RBUF 
        (f) if ga modulo is == 0:
            • Save ANET’s current parameters for later use in tournament play.
        """


