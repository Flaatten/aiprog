import os

from ANETCase import ANETCase
from ANETDataset import ANETDataset
from ActorNet import ActorNet
from Hex import Hex
from MonteCarloTreeSearch import MonteCarloTreeSearch
from ReplayBuffer import ReplayBuffer
from TOPP import TOPP
from constant.ANETOptimizer import ANETOptimizer
from constant.ActivationFunction import ActivationFunction
from state.InitialStateValidator import InitialStateValidator
import torch

import copy


def test():
    size = 3
    number_of_episodes = 1000
    number_of_search_games_per_actual_move = 30
    learning_rate = 0.04
    hidden_layers_structure = [7, 5]
    hidden_nodes_activation_function = ActivationFunction.TANH
    ANET_optimizer = ANETOptimizer.RMSPROP
    number_of_ANETs_to_cache_for_TOPP = 1
    num_games_to_be_played_between_two_agenst_during_TOPP = 2
    save_interval_for_ANET = 50
    number_of_cases_from_buffer_for_intertraining = 20
    epsilon = 0.3
    starting_player = 0

    buffer = ReplayBuffer()
    net = ActorNet(input_dim=(size ** 2) * 2 + 2, hidden_layers=hidden_layers_structure, output_dim=size ** 2, learning_rate=learning_rate, optimizer=ANET_optimizer,
                   hidden_nodes_activation_function=hidden_nodes_activation_function)

    for n in range(number_of_episodes):
        print(n)
        game = Hex("01" + "00" * (size ** 2), starting_player, number_of_search_games_per_actual_move)  # TODO HAVE A LOOK AT THE FIRST INPUT ARGUMENT HERE
        # game = Hex("00" * (size ** 2), 0, number_of_episodes)
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
            # net.assess_performance(num_games=10, starting_player=starting_player, size=size, number_of_search_games_per_actual_move=num_games_to_be_played_between_two_agenst_during_TOPP)
            path = ""
            path += "size:" + str(size)
            torch.save(net, "size")


def run_sim():
    sizes = [3, 4, 5, 6]
    num_episodes = [301]
    num_search_games_per_actual_move = [25, 50]
    hidden_layers = []

    for i in range(3, 10, 2):
        for j in range(2, 8, 2):
            hidden_layers.append[i, j]

    for size in sizes:
        for episode in num_episodes:
            for num_search_games in num_search_games_per_actual_move:
                for layers in hidden_layers:
                    run_with_config(size, episode, num_search_games, layers)

def run_sim_using_best_configs(k):
    if k == 3:
        run_with_config(k, 401, 50, [9, 2])
    elif k == 4:
        run_with_config(k, 401, 50, [5, 6])
    elif k == 5:
        run_with_config(k, 401, 50, [9, 6])
    elif k == 6:
        run_with_config(k, 401, 50, [7, 4])



def run_with_config(size, num_episodes, num_search_games_per_actual_move, hidden_layers, save_interval=50):
    number_of_episodes = num_episodes
    number_of_search_games_per_actual_move = num_search_games_per_actual_move
    learning_rate = 0.04
    hidden_layers_structure = hidden_layers
    hidden_nodes_activation_function = ActivationFunction.RELU
    ANET_optimizer = ANETOptimizer.SGD
    number_of_ANETs_to_cache_for_TOPP = 1
    num_games_to_be_played_between_two_agenst_during_TOPP = 2
    save_interval_for_ANET = save_interval
    number_of_cases_from_buffer_for_intertraining = 20
    epsilon = 0.3
    starting_player = 0

    buffer = ReplayBuffer()
    net = ActorNet(input_dim=(size ** 2) * 2 + 2, hidden_layers=hidden_layers_structure, output_dim=size ** 2, learning_rate=learning_rate, optimizer=ANET_optimizer,
                   hidden_nodes_activation_function=hidden_nodes_activation_function)

    for n in range(num_episodes):
        print(n)
        game = Hex("01" + "00" * (size ** 2), starting_player, number_of_search_games_per_actual_move)  # TODO HAVE A LOOK AT THE FIRST INPUT ARGUMENT HERE
        # game = Hex("00" * (size ** 2), 0, number_of_episodes)
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
            # net.assess_performance(num_games=10, starting_player=starting_player, size=size, number_of_search_games_per_actual_move=num_games_to_be_played_between_two_agenst_during_TOPP)
            path = ""
            path += "size:" + str(size)
            path += "num_ep:" + str(num_episodes)
            path += "num_search:" + str(num_search_games_per_actual_move)
            path += "hidden_layers:["

            for layer in hidden_layers:
                path += str(layer) + ","
            path += "]"

            path += "iteration:" + str(n)

            torch.save(net, path + ".model")

def test_topp(n): # use for point 5 of requirements
    paths = TOPP.get_model_paths(n)
    print(paths)

    topp = TOPP(n, 20, paths, verbose=False)
    topp.play_tournament()

def short_training_session_plus_topp(k, verbose=False):
    number_of_episodes = 15
    number_of_search_games_per_actual_move = 10
    learning_rate = 0.1

    if k == 3:
        hidden_layers_structure = [9, 2]
    elif k == 4:
        hidden_layers_structure = [5, 6]
    elif k == 5:
        hidden_layers_structure = [9, 6]
    elif k == 6:
        hidden_layers_structure = [7, 4]

    hidden_nodes_activation_function = ActivationFunction.RELU
    ANET_optimizer = ANETOptimizer.SGD
    number_of_ANETs_to_cache_for_TOPP = 1
    num_games_to_be_played_between_two_agenst_during_TOPP = 2
    save_interval_for_ANET = 3
    number_of_cases_from_buffer_for_intertraining = 20
    epsilon = 0.3
    starting_player = 0

    cached_models = []
    buffer = ReplayBuffer()
    net = ActorNet(input_dim=(k ** 2) * 2 + 2, hidden_layers=hidden_layers_structure, output_dim=k ** 2, learning_rate=learning_rate, optimizer=ANET_optimizer,
                   hidden_nodes_activation_function=hidden_nodes_activation_function)

    for n in range(number_of_episodes):
        game = Hex("01" + "00" * (k ** 2), starting_player, number_of_search_games_per_actual_move)  # TODO HAVE A LOOK AT THE FIRST INPUT ARGUMENT HERE
        # game = Hex("00" * (size ** 2), 0, number_of_episodes)
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
            cached_models.append(copy.deepcopy(net))
            print(len(cached_models))

    topp = TOPP(k, 20, cached_models, verbose=verbose)
    topp.play_tournament()


#short_training_session_plus_topp(4, verbose=False)
test_topp(5)