import jax
import jax.numpy as jnp
import jax.scipy.signal as signal
from random import choice
from typing import Self, override
from mcts import MCTS, State
from dataclasses import dataclass


horizontal_kernel = jnp.array([[1, 1, 1]])
vertical_kernel = jnp.transpose(horizontal_kernel)
diag1_kernel = jnp.eye(3, dtype=jnp.uint8)
diag2_kernel = jnp.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]


class TicTacToeState(State[jax.Array]):
    """State of the game"""

    actor: int
    board: jax.Array

    def __init__(self, actor: int, board: jax.Array):
        self.actor = actor
        self.board = board
    
    
    def next_actor(self: Self) -> int:
        return 1 if self.actor == 2 else 2
    
    @override
    def legal_actions(self: Self) -> list[jax.Array]:
        return [a for a in jnp.argwhere(self.board == 0)]

    @override
    def act(self: Self, action: jax.Array) -> tuple[Self, float, bool]:
        # modify board
        new_board = self.board.at[action[0], action[1]].set(self.actor)
        new_actor = self.next_actor()
        new_state = type(self)(new_actor, new_board)
        
        def won(board: jax.Array, actor: int) -> bool:
            obs = board == actor
            for kernel in detection_kernels:
                if (signal.convolve2d(obs, kernel, mode="valid") == 3).any().item():
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

    @override
    def display(self: Self, indent) -> str:
        lines = str(self.board).splitlines()
        return "\n".join(indent + line for line in lines)

def new_tic_tac_toe_board():
    return TicTacToeState(
        actor=1,
        board=jnp.zeros((3, 3), dtype=jnp.uint8, device=jax.devices("cpu")[0])
    )


if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'cpu')
    # initialize mcts engine with root state
    mcts = MCTS(new_tic_tac_toe_board())
    # rollout
    for i in range(1000):
        # choose best action
        print("Rollout", i)
        mcts.rollout(mcts.root)
        
    mcts.print_tree()


    action = mcts.choose(mcts.root)
    print(action)
