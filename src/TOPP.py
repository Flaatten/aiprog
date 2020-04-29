from state.HexGameState import HexGameState


class TOPP:

    def __init__(self, board_size, series_size, paths, verbose=False):

        self.models = []
        self.number_of_policies = len(paths)
        self.board_size = board_size
        self.series_size = series_size
        self.scores = []
        self.verbose = verbose

        for i in range(len(paths)):
            self.models.append(torch.load(paths[i]))

    def play_tournament(self):

        print("Playing tournament...")

        for policy_i in range(self.number_of_policies):
            for policy_j in range(self.number_of_policies):
                if policy_j > policy_i:
                    self.play_series(policy_i, policy_j)

        for i in range(self.number_of_policies):
            print("Policy number " + str(i) + " won " + str(self.scores[i]) + " out of " + str(self.series_size) + " matches")



    def play_series(self, policy_i, policy_j):

        for game_number in range(self.series_size):
            if game_number%2 == 0:
                self.play_game(self, policy_i, policy_j)
            else:
                self.play_game(self, policy_j, policy_i)

    def play_game(self, player_1, player_2):

        policies = [player_1, player_2]
        move_number = 0
        player_to_move = 0
        board = self._create_board()
        game_state = HexGameState(board, 0, None)

        while game_state.is_game_over != True:

            if self.verbose == True:
                game_state.render()

            current_player = game_state.next_to_move
            action = self.models[policies[current_player]].predict(game_state,0)
            game_state = HexGameState.move()

        if self.verbose == True:
            game_state.render()

        winner = HexGameState.game_result[1]
        self.scores[policies[winner]] += 1

    def _create_board(self):
        board = [[[0,0] for j in range(self.board_size)] for i in range(self.board_size)]
        return board
