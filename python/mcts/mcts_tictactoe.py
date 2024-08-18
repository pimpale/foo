"""
An example implementation of the abstract Node class for use in MCTS
If you run this file then you can play against the computer.
A tic-tac-toe board is represented as a tuple of 9 values, each either None,
True, or False, respectively meaning 'empty', 'X', and 'O'.
The board is indexed by row:
0 1 2
3 4 5
6 7 8
For example, this game board
O - X
O X -
X - -
corrresponds to this tuple:
(False, None, True, False, True, None, True, None, None)
"""

import jax
import jax.numpy as jnp
import jax.scipy.signal as signal
from random import choice
from typing import Self, override
from mcts import MCTS, StateBase
from dataclasses import dataclass


horizontal_kernel = jnp.array([[1, 1, 1]])
vertical_kernel = jnp.transpose(horizontal_kernel)
diag1_kernel = jnp.eye(3, dtype=jnp.uint8)
diag2_kernel = jnp.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]


class State(StateBase):
    """State of the game"""

    actor: int
    board: jax.Array

    def __init__(self, actor: int, board: jax.Array):
        self.actor = actor
        self.board = board
    
    
    def next_actor(self: Self) -> int:
        return 1 if self.actor == 2 else 2
    
    @override
    def legal_actions(self: Self) -> jax.Array:
        return jnp.argwhere(self.board == 0)[0]

    @override
    def act(self: Self, action: jax.Array) -> tuple[Self, float, bool]:
        # modify board
        new_board = self.board.at[action].set(self.actor)
        new_actor = self.next_actor()
        new_state = type(self)(new_actor, new_board)
        
        def won(board: jax.Array, actor: int) -> bool:
            obs = board == actor
            for kernel in detection_kernels:
                if signal.convolve2d(obs, kernel, mode="valid") == 3:
                    return True
            return False
        
        if won(new_board, self.actor):
            return (new_state, 1.0, True)
        elif won(new_board, self.next_actor()):
            return (new_state, -1.0, True)
        elif (new_board != 0).all().item():
            return (new_state, 0.0, True)        
        else:
            return (new_state, 0.0, False)



class Observation:
    """Observation by a single player of the game"""

    board: jax.Array

    def __init__(self, state: State, actor: int):
        s = state.board
        self.board = jnp.stack(
            [s == actor, s != actor, s == 0],
        ).astype(jnp.uint8)

    def is_winner(self: Self) -> bool:
        for kernel in detection_kernels:
            if signal.convolve2d(self.board[0], kernel, mode="valid") == 3:
                return True
        return False

def play_game():
    tree = MCTS()
    board = new_tic_tac_toe_board()
    print(board.to_pretty_string())
    while True:
        row_col = input("enter row,col: ")
        row, col = map(int, row_col.split(","))
        index = 3 * (row - 1) + (col - 1)
        if board.tup[index] is not None:
            raise RuntimeError("Invalid move")
        board = board.make_move(index)
        print(board.to_pretty_string())
        if board.terminal:
            break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(50):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        if board.terminal:
            break


def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal



def new_tic_tac_toe_board():
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False)


if __name__ == "__main__":
    play_game()
