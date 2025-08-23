import numpy as np
import numpy.typing as npt
from scipy.signal import convolve2d
from typing import Any, TypeAlias, Literal

class State:
    """State of the game"""
    board: np.ndarray[Any, np.dtype[np.int8]]

    def __init__(self, board:np.ndarray[Any, np.dtype[np.int8]]):
        self.board = board

class Observation:
    """Observation by a single player of the game"""
    board: np.ndarray[Any, np.dtype[np.int8]]

    def __init__(self, board:np.ndarray[Any, np.dtype[np.int8]]):
        self.board = board

# Column to place it in
Action:TypeAlias = np.int8

# Reward for the agent
Reward:TypeAlias = np.float32

# Value of an observation for an agent
Value:TypeAlias = np.float32

# Advantage of a particular action for an agent
Advantage:TypeAlias = np.float32

Player:TypeAlias = np.int8

PLAYER1:Player = np.int8(1)
PLAYER2:Player = np.int8(2)

def opponent(actor:Player) -> Player:
    return PLAYER2 if actor == PLAYER1 else PLAYER1

def print_obs(o:Observation):
    for row in reversed(o.board):
        # We print '#' for our item, and 'O' for the opponent
        for x in row:
            c = ' '
            if x == 1:
                c = '#'
            elif x == 2:
                c = 'O'
            print(c, end=" ")
        print()
    print(flush=True)

def initial_state(dims:tuple[int, int]) -> State:
    return State(np.zeros(dims, dtype=np.int8))

def state_to_observation(state: State, actor: np.int8) -> Observation:
    s = state.board
    o = s.copy()
    o[s == actor] = 1
    o[s != actor] = 2
    o[s == 0] = 0
    return Observation(o)


horizontal_kernel = np.array([[ 1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

def is_winner(state:State, actor:Player) -> bool:
    board = state.board
    for kernel in detection_kernels:
        if (convolve2d(board == actor, kernel, mode="valid") == 4).any():
            return True
    return False

def drawn(state:State) -> bool:
    """
    Returns if the board is completely filled
    """
    return 0 not in state.board

# return the reward for the actor
def state_to_reward(s: State, player: Player) -> Reward:
    if is_winner(s, player):
        return np.float32(1.0)
    elif is_winner(s, opponent(player)):
        return np.float32(-1.0)
    else:
        return np.float32(0.0)

class Env():
    def __init__(
        self,
        dims:tuple[int,int]
    ):
        self._game_over = False
        self._winner = None
        self._moves = []
        self.state: State = initial_state(dims)

    def reset(self) -> None:
        self._game_over = False
        self._winner = None
        self._moves = []
        self.state = initial_state(self.state.board.shape)

    def observe(self, actor: np.int8) -> Observation:
        return state_to_observation(self.state, actor)

    def game_over(self) -> bool:
        return self._game_over

    def winner(self) -> Player | None:
        return self._winner

    def dims(self) -> tuple[int, int]:
        return self.state.board.shape

    def legal_mask(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        return self.state.board[-1] == 0

    def legal_actions(self) -> list[Action]:
        return [Action(i) for i in range(self.dims()[1]) if self.legal_mask()[i]]

    def step(self, a: Action, actor: Player) -> Reward:
        for i,row in enumerate(self.state.board):
            if row[a] == 0:
                self._moves.append((i,a))
                row[a] = actor
                break

        r = state_to_reward(self.state, actor)

        if r != 0:
            self._game_over = True
            self._winner = actor
        elif drawn(self.state):
            self._game_over = True

        return r
    
    def undo(self):
        if len(self._moves) == 0:
            return
        self.state.board[self._moves.pop()] = 0
        self._winner = None
        self._game_over = False
