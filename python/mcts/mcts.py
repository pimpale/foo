"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Optional, Self
import typing

import jax


# ```java
# // generics in java
# class StateBase<Action> {
# ```


class State[Action](ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    action_type: type[Action]

    @abstractmethod
    def legal_actions(self: Self) -> list[Action]:
        "All possible successors of this board state"
        raise NotImplementedError

    @abstractmethod
    def act(self: Self, action: Action) -> tuple[Self, float, bool]:
        "Returns next state, reward and whether the game ends"
        raise NotImplementedError


class MCTS[Action]:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    @dataclass
    class Node:
        state: State[Action]  # board state
        reward: float  # reward gained by reaching this node
        terminal: bool  # whether this node is terminal
        visit_count: int  # number of times this node was visited
        unvisited_actions: list[Action]  # untried actions
        children: list[tuple[Action, int]]  # (action, child_node) pairs

    rng: jax.Array
    nodes: list[Node]  # all nodes
    exploration_weight: float

    def __init__(self: Self, exploration_weight=1):
        self.rng = jax.random.key(0)
        self.nodes = []
        self.exploration_weight = exploration_weight

    def choose(self: Self) -> Action:
        "Choose the best successor of node. (Choose a move in the game)"

        node = self.nodes[0]

        if node.terminal:
            raise RuntimeError(f"choose called on terminal node")

        # if we don't have any children, just act randomly
        if not node.children:
            key1, key2 = jax.random.split(self.rng)
            self.rng = key1
            legal_actions = node.state.legal_actions()
            return legal_actions[
                jax.random.randint(key2, shape=(), minval=0, maxval=len(legal_actions))
            ]

        def score(action_child: tuple[jax.Array, int]):
            _, n = action_child
            if self.nodes[n].visit_count == 0:
                return float("-inf")  # avoid unseen moves
            return (
                self.nodes[n].total_reward / self.nodes[n].visit_count
            )  # average reward

        action, _ = max(node.children, key=score)

        return action

    def do_rollout(self: Self, node_idx: int):
        "Make the tree one layer better. (Train for one iteration.)"
        path, maybe_action = self._select(node_idx)
        if maybe_action is not None:
            bud_idx = path[-1]
            self._expand(bud_idx, maybe_action)
            reward = self._simulate(bud_idx)
            self._backpropagate(path, reward)

    def _select(self: Self, node_idx: int) -> tuple[list[int], Optional[Action]]:
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node_idx)
            node = self.nodes[node_idx]
            if node.terminal:
                return path, None
            elif len(node.unvisited_actions) > 0:
                action = node.unvisited_actions.pop()
                return path, action
            else:
                node_idx = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node_idx: int, maybe_action: Action):
        "Update the `children` dict with the children of `node`"
        state, reward, terminal = self.nodes[node_idx].state.act(maybe_action)
        self.nodes.append(
            self.Node(
                state=state,
                reward=reward,
                terminal=terminal,
                visit_count=0,
                unvisited_actions=state.legal_actions(),
                children=[],
            )
        )

    def _simulate(self: Self, node: Node) -> float:
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            if node.terminal:
                reward = node.reward
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self: Self, path: list[State], reward: float):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self: Self, node: Node) -> int:
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert len(node.unvisited_actions) == 0

        log_N_vertex = math.log(node.visit_count)

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
