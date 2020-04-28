from src.action.HexMove import HexMove
import numpy as np


class HexGameState:

    def __init__(self, state, next_to_move, action_that_resulted_in_the_current_state):
        self.board = state
        self.next_to_move = next_to_move
        self.action_that_resulted_in_the_current_state = action_that_resulted_in_the_current_state
        self.result = [0, 0]
        self.directions = np.array([[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]])

    def get_board(self):
        return self.board

    def get_action_that_resulted_in_the_current_state(self):
        return self.action_that_resulted_in_the_current_state

    def game_result(self):
        return self.result

    def is_game_over(self):

        # PLAYER 0:

        path = {}
        for i in range(len(self.board)):
            path[i] = {}
            for j in range(len(self.board)):
                path[i][j] = False

        for col_j in range(len(self.board)):
            if self.board[0][col_j][0] == 1:
                path[0][col_j] = True
                if self.add_to_path(path, [0, col_j], 0):
                    self.result[0] = 1
                    return True
        # PLAYER 1:

        for i in range(len(self.board)):
            for j in range(len(self.board)):
                path[i][j] = False

        for row_i in range(len(self.board)):
            if self.board[row_i][0][1] == 1:
                path[row_i][0] = True
                if self.add_to_path(path, [row_i, 0], 1):
                    self.result[1] = 1
                    return True

        return False

    def add_to_path(self, path, cell, player):
        for neighbour in self.get_neighbour_cells(cell):
            n_row = neighbour[0]
            n_col = neighbour[1]
            if path[n_row][n_col] != True and self.board[n_row][n_col][player] == 1:
                if (player == 0 and n_row == len(self.board) - 1) or (player == 1 and n_col == len(self.board) - 1):
                    return True
                else:
                    path[n_row][n_col] = True
                    if self.add_to_path(path, [n_row, n_col], player) == True:
                        return True

    def get_neighbour_cells(self, cell):
        neighbours = np.add(np.array([cell, cell, cell, cell, cell, cell]), self.directions)
        return self.remove_illegal_neighbours(neighbours)

    def remove_illegal_neighbours(self, neighbours):
        neighbour_list = []
        for neighbour in neighbours:
            n_row = neighbour[0]
            n_col = neighbour[1]
            if n_row >= 0 and n_row < len(self.board) and n_col >= 0 and n_col < len(self.board):
                neighbour_list.append(neighbour)
        return neighbour_list

    def is_move_legal(self, move):
        return self._is_open_cell([move.row, move.col])

    def move(self, action):
        row = action.row
        col = action.col
        player = action.player
        _board = self.board
        _board[row][col][player] = 1
        return HexGameState(_board, 1 - player, action)  # TODO FINISH

    def get_legal_actions(self):
        legal_actions = []
        for row_i in range(len(self.board)):
            for col_j in range(len(self.board)):
                if self._is_open_cell([row_i, col_j]):
                    legal_actions.append(HexMove(row_i, col_j, self.next_to_move))
        return legal_actions

    def print_move(self):
        pass

    def _is_open_cell(self, cell):
        first_coordinate = cell[0]
        second_coordiate = cell[1]
        board_cell = self.board[first_coordinate][second_coordiate]
        return board_cell[0] + board_cell[1] < 1

    def to_string(self):
        p1_turn = str(self.next_to_move - 1)
        p2_turn = str(self.next_to_move)
        stateString = p1_turn + p2_turn
        for row_i in range(len(self.board)):
            for col_j in range(len(self.board)):
                for player in range(2):
                    stateString += self.board[row_i][col_j][player]

    def _render_cell_value(self, row, col):
        if self.board[row][col][0] == 1:
            return "(X)"
        elif self.board[row][col][1] == 1:
            return "(O)"
        else:
            return "( )"

    def render(self):
        print()
        print("Player1 = X,  Player 2 = O")
        for i in range(len(self.board)):
            string = "  " * (len(self.board) - i)
            for j in range(i + 1):
                string += " " + self._render_cell_value(i - j, j)
            print(string)
        for i in range(1, len(self.board)):
            string = "  " * (i + 1)
            for j in range(len(self.board) - i):
                string += " " + self._render_cell_value(len(self.board) - 1 - j, i + j)
            print(string)







