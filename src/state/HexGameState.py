class HexGameState:

    def __init__(self, state, next_to_move, action_that_resulted_in_the_current_state):
        self.board = state
        self.next_to_move = next_to_move
        self.action_that_resulted_in_the_current_state = action_that_resulted_in_the_current_state

    def get_board(self):
        return self.board

    def get_action_that_resulted_in_the_current_state(self):
        return self.action_that_resulted_in_the_current_state

    def game_result(self):
        pass

    def is_game_over(self):
        pass

    def is_move_legal(self, move):
        pass

    def move(self, action):
        return HexGameState() # TODO FINISH

    def get_legal_actions(self):
        pass

    def print_move(self):
        pass



    def to_string(self):
        pass

