#%%
#% load_ext autoreload
#% autoreload 2
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from typing import Self, override
from mcts import SinglePlayerMCTS, State
from dataclasses import dataclass


horizontal_kernel = jnp.array([[1, 1, 1]])
vertical_kernel = jnp.transpose(horizontal_kernel)
diag1_kernel = jnp.eye(3, dtype=jnp.uint8)
diag2_kernel = jnp.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]



class TicTacToeState(State[jax.Array]):
    """State of the game"""

    board: jax.Array

    def __init__(self, actor: int, board: jax.Array):
        self.actor = actor
        self.board = board

    def next_actor(self: Self) -> int:
        return 1 if self.actor == 2 else 2

    @override
    def legal_actions(self: Self) -> list[jax.Array]:
        return [a for a in jnp.argwhere(self.board == 0)]

    def _act(self: Self, action: jax.Array) -> tuple[Self, float, bool]:
        # modify board
        new_board = self.board.at[action[0], action[1]].set(self.actor)
        new_actor = self.next_actor()
        new_state = type(self)(new_actor, new_board)

        def won(board: jax.Array, actor: int) -> bool:
            obs = board == actor
            for kernel in detection_kernels:
                if (convolve2d(obs, kernel, mode="valid") == 3).any().item():
                    return True
            return False

        if won(new_board, self.actor):
            return (new_state, 1.0, True)
        elif (new_board != 0).all().item():
            return (new_state, 0.0, True)
        else:
            return (new_state, 0.0, False)


    @staticmethod
    def _heuristic(board: jax.Array, me: int, opp: int) -> float:
        my_score:float = 0
        opp_score:float = 0
        for kernel in detection_kernels:
            my_score += jnp.sum(convolve2d(board == me, kernel, mode="valid") >= 2).item()
            opp_score += jnp.sum(convolve2d(board == opp, kernel, mode="valid") >= 2).item()
            
        return my_score - opp_score
    
    def _heuristic_player(self: Self) -> jax.Array:
        best_action = None
        best_score = float("-inf")
        for action in self.legal_actions():
            board = self.board.at[action[0], action[1]].set(self.actor)
            score = self._heuristic(board, self.actor, self.next_actor())
            if score > best_score:
                best_score = score
                best_action = action
        
        assert best_action is not None
        return best_action
    
    def _naive_player(self: Self) -> jax.Array:
        return self.legal_actions()[0]

    @override
    def act(self: Self, action: jax.Array) -> tuple[Self, float, bool]:
        # take player action
        s, r, t = self._act(action)
        # take random action
        if not t:
            s, r, t = s._act(s._heuristic_player())
            r = -r

        return s, r, t

    @override
    def display(self: Self, indent) -> str:
        lines = str(self.board).splitlines()
        return "\n".join(indent + line for line in lines)



def new_tic_tac_toe_board():
    return TicTacToeState(
        actor=1, board=jnp.zeros((3, 3), dtype=jnp.uint8, device=jax.devices("cpu")[0])
    )

#%%

jax.config.update("jax_platform_name", "cpu")

# initialize mcts engine with root state
mcts = SinglePlayerMCTS(new_tic_tac_toe_board())
# rollout
for i in range(1000):
    # choose best action
    print("Rollout", i)
    mcts.rollout(mcts.root)

#%%
mcts.print_tree()

#%%
action = mcts.choose(mcts.root)
print(action)

# %%
