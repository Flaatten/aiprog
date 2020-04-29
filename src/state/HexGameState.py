import copy
import random

import numpy as np

from action.HexMove import HexMove


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
        return self.is_open_cell([move.row, move.col])

    def move(self, action):
        row = int(action.row)
        col = int(action.col)
        player = action.player
        _board = copy.deepcopy(self.board)
        _board[row][col][player] = 1
        return HexGameState(_board, 1 - player, action)

    def get_legal_actions(self):
        legal_actions = []
        for row_i in range(len(self.board)):
            for col_j in range(len(self.board)):
                if self.is_open_cell([row_i, col_j]):
                    legal_actions.append(HexMove(row_i, col_j, self.next_to_move))
        return legal_actions

    def is_open_cell(self, cell):
        first_coordinate = cell[0]
        second_coordiate = cell[1]
        board_cell = self.board[first_coordinate][second_coordiate]
        return board_cell[0] + board_cell[1] < 1

    def set_invalid_actions_to_zero_from_list(self, list):
        size = np.sqrt(len(list))
        for i in range(len(list)):
            row_index = int(np.floor(i / size))
            col_index = int(i % size)

            cell = [row_index, col_index]

            #if list[i] < 0:
            #    list[i] = 0

            if not self.is_open_cell(cell):
                list[i] = 0
            else:
                pass
        return list


    def to_string(self):
        p1_turn = str(1 - self.next_to_move)
        p2_turn = str(self.next_to_move)
        stateString = ""
        stateString += p1_turn + p2_turn
        for row_i in range(len(self.board)):
            for col_j in range(len(self.board)):
                for player in range(2):
                    stateString += str(self.board[row_i][col_j][player])

        return stateString

    # used as input for neural net
    def as_list(self):
        return [int(i) for i in self.to_string()]

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

    def get_greedy_move(self, distribution):
        current_best_index = None
        current_best_score = None
        for i in range(len(distribution)):
            if current_best_score is None:
                current_best_index = i
                current_best_score = distribution[i]
            else:
                score = distribution[i]

                if score > current_best_score:
                    current_best_index = i
                    current_best_score = distribution[i]

        if current_best_index is None:
            raise ValueError("Distribution empty")

        num_rows = np.sqrt(len(distribution))
        row_index = np.floor(current_best_index / num_rows)
        col_index = current_best_index % num_rows

        move = HexMove(row_index, col_index, self.next_to_move)
        return move

    def get_random_weighted_move(self, distribution):
        rand_num = random.random()

        sum_ = 0

        for i in range(len(distribution)):
            sum_ += distribution[i]

            if sum_ >= rand_num:
                move_index = i
                # move index i = the move to be taken
                # find row and col
                num_rows = len(self.board)

                row_index = np.floor(move_index / num_rows)
                col_index = move_index % num_rows

                move = HexMove(row_index, col_index, self.next_to_move)
                return move

        raise ValueError("No move found, using the distribution and random number " + str(rand_num))

        # move i is the next move