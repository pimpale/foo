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


@dataclass
class Node[Action]:
    state: State[Action]  # board state
    reward: float  # reward gained by reaching this node
    children_reward: float  # reward gained by all its children
    terminal: bool  # whether this node is terminal
    visit_count: int  # number of times this node was visited
    unvisited_actions: list[Action]  # untried actions
    children: list[tuple[Action, Self]]  # (action, child_node) pairs


class MCTS[Action]:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    rng: jax.Array
    root: Optional[Node[Action]]
    exploration_weight: float

    def __init__(self: Self, exploration_weight=1):
        self.rng = jax.random.key(0)
        self.nodes = []
        self.exploration_weight = exploration_weight

    def randint(self: Self, low: int, high: int) -> int:
        key1, key2 = jax.random.split(self.rng)
        self.rng = key1
        return jax.random.randint(key2, shape=(), minval=low, maxval=high).item()

    def choose(self: Self) -> Action:
        "Choose the best successor of node. (Choose a move in the game)"

        node = self.nodes[0]

        if node.terminal:
            raise RuntimeError(f"choose called on terminal node")

        # if we don't have any children, just act randomly
        if not node.children:
            legal_actions = node.state.legal_actions()
            return legal_actions[self.randint(0, len(legal_actions))]

        def score(action_child: tuple[jax.Array, int]):
            _, n = action_child
            if self.nodes[n].visit_count == 0:
                return float("-inf")  # avoid unseen moves
            return (
                self.nodes[n].total_reward / self.nodes[n].visit_count
            )  # average reward

        action, _ = max(node.children, key=score)

        return action

    def do_rollout(self: Self, node: Node):
        "Make the tree one layer better. (Train for one iteration.)"
        path, maybe_action = self._select(node)
        if maybe_action is not None:
            bud = path[-1]
            self._expand(bud, maybe_action)
            reward = self._simulate(bud)
            self._backpropagate(path, reward)

    def _select(self: Self, node: Node) -> tuple[list[Node], Optional[Action]]:
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node.terminal:
                return path, None
            elif len(node.unvisited_actions) > 0:
                action = node.unvisited_actions.pop()
                return path, action
            else:
                node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node: Node, action: Action):
        "Update the `children` dict with the children of `node`"
        state, reward, terminal = node.state.act(action)
        self.nodes.append(
            Node(
                state=state,
                reward=reward,
                children_reward=0,
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
                return - reward if invert_reward else reward

            action, node = node.children[self.randint(0, len(node.children))]
            invert_reward = not invert_reward

    def _backpropagate(self: Self, path: list[Node], reward: float):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            node.visit_count += 1
            node.children_reward += reward
            reward = -reward  # 1 for me is -1 for you

    def _uct_select(self: Self, node: Node) -> Node:
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert len(node.unvisited_actions) == 0

        log_N_vertex = math.log(node.visit_count)

        def uct(action_child: tuple[Action, Node]) -> float:
            action, node = action_child
            "Upper confidence bound for trees"
            return (
                node.visit_count / node.children_reward
                + self.exploration_weight * math.sqrt(log_N_vertex / node.visit_count)
            )

        return max(node.children, key=lambda action_child: uct(action_child))[1]
