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
    size = 4
    number_of_episodes = 1000
    number_of_search_games_per_actual_move = 50
    learning_rate = 0.05
    hidden_layers_structure = [7, 6]
    hidden_nodes_activation_function = ActivationFunction.RELU
    ANET_optimizer = ANETOptimizer.SGD
    number_of_ANETs_to_cache_for_TOPP = 1
    num_games_to_be_played_between_two_agenst_during_TOPP = 2
    save_interval_for_ANET = 5
    number_of_cases_from_buffer_for_intertraining = 20
    epsilon = 0.3
    starting_player = 0

    buffer = ReplayBuffer()
    net = ActorNet(input_dim=(size ** 2) * 2 + 2, hidden_layers=hidden_layers_structure, output_dim=size ** 2, learning_rate=learning_rate, optimizer=ANET_optimizer, hidden_nodes_activation_function=hidden_nodes_activation_function)

    for n in range(number_of_episodes):
        print(n)
        game = Hex("01" + "00" * (size ** 2), starting_player, number_of_search_games_per_actual_move)  # TODO HAVE A LOOK AT THE FIRST INPUT ARGUMENT HERE
        #game = Hex("00" * (size ** 2), 0, number_of_episodes)
        if InitialStateValidator.is_valid(game.root.state.board):
            while not game.is_finished():
                s_init = copy.deepcopy(game.root)
                mcts = MonteCarloTreeSearch(s_init, net, epsilon)
                mcts.best_action(number_of_search_games_per_actual_move)

                distribution = mcts.root.get_standardised_distribution()

                buffer.add_case(ANETCase(mcts.root, distribution))

                move = s_init.state.get_random_weighted_move(distribution)
                game.move(move)

            train_data = ANETDataset(buffer.get_mini_batch(number_of_cases_from_buffer_for_intertraining))  # list of nodes
            net.train_using_buffer_subset(train_data)


        else:
            raise ValueError("State not valid: " + game.root.state)

        if n % 5 == 0:
            epsilon = epsilon * 0.985
            learning_rate = learning_rate * 0.985

        if n % save_interval_for_ANET == 0:
            net.assess_performance(num_games=10, starting_player=starting_player, size=size, number_of_search_games_per_actual_move=num_games_to_be_played_between_two_agenst_during_TOPP)
            #net.save_params(n)



test()