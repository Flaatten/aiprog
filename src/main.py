from Hex import Hex
from ReplayBuffer import ReplayBuffer
from constant.ANETOptimizer import ANETOptimizer
from constant.ActivationFunction import ActivationFunction
from state.InitialStateValidator import InitialStateValidator


def test():
    size = 3
    number_of_episodes = 10
    number_of_search_games_per_actual_mvoe = 100
    learning_rate = 0.01
    hidden_layers_structure = [5, 5, 5]
    activation_function = ActivationFunction.RELU
    ANET_optimizer = ANETOptimizer.SGD
    number_of_ANETs_to_cache_for_TOPP = 1
    num_games_to_be_played_between_two_agenst_during_TOPP = 2
    save_interval_for_ANET = 40

    game = Hex()

    if (InitialStateValidator.is_valid(game.get_state())):
        game.run()