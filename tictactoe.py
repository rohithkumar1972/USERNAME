"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)

    return X if x_count <= o_count else O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return {(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY}

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise Exception("Invalid action")

    new_board = [row.copy() for row in board]
    new_board[action[0]][action[1]] = player(board)
    return new_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows, columns, and diagonals
    for i in range(3):
        if all(board[i][j] == X for j in range(3)):
            return X
        if all(board[j][i] == X for j in range(3)):
            return X
    if all(board[i][i] == X for i in range(3)) or all(board[i][2 - i] == X for i in range(3)):
        return X

    for i in range(3):
        if all(board[i][j] == O for j in range(3)):
            return O
        if all(board[j][i] == O for j in range(3)):
            return O
    if all(board[i][i] == O for i in range(3)) or all(board[i][2 - i] == O for i in range(3)):
        return O

    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return winner(board) is not None or all(all(cell != EMPTY for cell in row) for row in board)

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_result = winner(board)
    if winner_result == X:
        return 1
    elif winner_result == O:
        return -1
    else:
        return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    current_player = player(board)
    if current_player == X:
        return max_value(board)[1]
    else:
        return min_value(board)[1]

def max_value(board):
    if terminal(board):
        return utility(board), None

    v = float("-inf")
    move = None

    for action in actions(board):
        result_value, _ = min_value(result(board, action))
        if result_value > v:
            v = result_value
            move = action

    return v, move

def min_value(board):
    if terminal(board):
        return utility(board), None

    v = float("inf")
    move = None

    for action in actions(board):
        result_value, _ = max_value(result(board, action))
        if result_value < v:
            v = result_value
            move = action

    return v, move
