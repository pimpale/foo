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

    @abstractmethod
    def display(self: Self, indent: str) -> str:
        "Display the current state"
        raise NotImplementedError


@dataclass
class Node[Action]:
    state: State[Action]  # board state
    terminal: bool  # whether this node is terminal
    unvisited_actions: list[Action]  # untried actions
    children: list["Edge"]  # (action, child_node) pairs


@dataclass
class Edge[Action]:
    action: Action
    reward: float  # reward at this edge
    Q: float  # cumulative reward gained by this node + all its children
    N: int  # visit count
    node: Node[Action]


class SinglePlayerMCTS[Action]:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    rng: jax.Array
    root: Node[Action]
    exploration_weight: float

    def __init__(self: Self, root: State[Action], exploration_weight: float = 1):
        self.rng = jax.random.key(0)
        self.root = Node(
            state=root,
            terminal=False,
            unvisited_actions=root.legal_actions(),
            children=[],
        )
        self.exploration_weight = exploration_weight

    def randint(self: Self, low: int, high: int) -> int:
        key1, key2 = jax.random.split(self.rng)
        self.rng = key1
        return jax.random.randint(key2, shape=(), minval=low, maxval=high).item()

    def choose(self: Self, node: Node) -> Action:
        "Choose the best successor of node. (Choose a move in the game)"

        if node.terminal:
            raise RuntimeError(f"choose called on terminal node")

        # if we don't have any children, just act randomly
        if not node.children:
            legal_actions = node.state.legal_actions()
            return legal_actions[self.randint(0, len(legal_actions))]

        def score(edge: Edge) -> float:
            if edge.N == 0:
                return float("-inf")  # avoid unseen moves
            return edge.Q / edge.N  # average reward

        edge = max(node.children, key=score)

        return edge.action

    def rollout(self: Self, node: Node):
        "Make the tree one layer better. (Train for one iteration.)"
        path, bud, maybe_action_id = self._select(node)
        print("select_id", maybe_action_id)

        if maybe_action_id is not None:
            # expand node
            action = bud.unvisited_actions.pop(maybe_action_id)
            child = self._expand(bud, action)
            bud.children.append(child)
            # simulate
            reward = self._simulate(child)
            path.append(child)
            # backpropagate
            self._backpropagate(path, reward)
        else:
            # visit nodes
            self._backpropagate(path, 0)

    def _select(self: Self, node: Node) -> tuple[list[Edge], Node, Optional[int]]:
        "Find an unexplored descendent of `node`"
        path: list[Edge] = []
        while True:
            if node.terminal:
                return path, node, None
            elif len(node.unvisited_actions) > 0:
                action_id = self.randint(0, len(node.unvisited_actions))
                return path, node, action_id
            else:
                edge = self._uct_select(node)  # descend a layer deeper
                path.append(edge)
                print("_select", edge.action)
                node = edge.node

    def _expand(self, node: Node, action: Action):
        "Update the `children` dict with the children of `node`"
        state, reward, terminal = node.state.act(action)
        return Edge(
            action=action,
            reward=reward,
            Q=reward,
            N=1,
            node=Node(
                state=state,
                terminal=terminal,
                unvisited_actions=state.legal_actions(),
                children=[],
            ),
        )

    def _simulate(self: Self, edge: Edge) -> float:
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            if edge.node.terminal:
                reward = edge.reward
                return reward

            edge = self._expand(edge.node, edge.node.unvisited_actions[self.randint(0, len(edge.node.unvisited_actions))])

    def _backpropagate(self: Self, path: list[Edge], reward: float):
        "Send the reward back up to the ancestors of the leaf"
        for edge in reversed(path):
            edge.N += 1
            edge.Q += reward

    def _uct_select(self: Self, node: Node) -> Edge:
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert not node.terminal
        assert len(node.unvisited_actions) == 0

        N_s = sum(edge.N for edge in node.children)
        print("N_s", N_s)

        def uct(edge: Edge) -> float:
            "Upper confidence bound for trees"
            return edge.Q / edge.N + self.exploration_weight * math.sqrt(
                math.log(N_s) / edge.N
            )

        return max(node.children, key=uct)

    def print_tree(self: Self):
        self._print_tree(self.root, 0)
        
    def _print_tree(self: Self, node: Node, depth: int):
        print(f"{'\t' * depth}<{depth}>")
        print(node.state.display('\t' * depth))
        if node.terminal:
            print(f"{'\t' * depth}terminal")
        else:
            for edge in node.children:
                print(f"{'\t' * depth}action: {edge.action}, reward: {edge.reward}, Q: {edge.Q}, N: {edge.N}")
                self._print_tree(edge.node, depth + 1)