from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional
import numpy as np
import math


class GreedyStochastic(BestFirstSearch):
    def __init__(self, heuristic_function_type: HeuristicFunctionType,
                 T_init: float = 1.0, N: int = 5, T_scale_factor: float = 0.95):
        # GreedyStochastic is a graph search algorithm. Hence, we use close set.
        super(GreedyStochastic, self).__init__(use_close=True)
        self.heuristic_function_type = heuristic_function_type
        self.T = T_init
        self.N = N
        self.T_scale_factor = T_scale_factor
        self.solver_name = 'GreedyStochastic (h={heuristic_name})'.format(
            heuristic_name=heuristic_function_type.heuristic_name)

    def _init_solver(self, problem: GraphProblem):
        super(GreedyStochastic, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        if self.close.has_state(successor_node.state):
            return

        if not self.open.has_state(successor_node.state):
            self.open.push_node(successor_node)

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        TODO: implement this method!
        Remember: `GreedyStochastic` is greedy.
        """

        return self.heuristic_function.estimate(search_node.state)

    def _get_probs(self, x_vector):
        """
        calculate probability vector based on current temperature.
        """
        alpha = np.min(x_vector)
        res = np.power(x_vector/alpha, -1 / self.T)
        sigma = np.sum(res)
        return res / sigma

    def _select_random_node(self, nodes):
        """
        return a random node based on a specific probability distribution.
        """
        x = np.array([node.expanding_priority for node in nodes])
        if np.min(x) == 0.0:
            return nodes[0] # Corner case, we have found a goal node
        else:
            probs = self._get_probs(x)
            return np.random.choice(nodes, p=probs)

    def _extract_next_search_node_to_expand(self) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         using the stochastic method to choose out of the N
         best items from open.
        Use `np.random.choice(...)` whenever you need to randomly choose
         an item from an array of items given a probabilities array `p`.
        You can read the documentation of `np.random.choice(...)` and
         see usage examples by searching it in Google.
        Notice: You might want to pop min(N, len(open) items from the
                `open` priority queue, and then choose an item out
                of these popped items. The other items have to be
                pushed again into that queue.
        """
        if self.open.is_empty():
            return None

        nodes = []
        for i in range(min(len(self.open), self.N)):
            nodes.append(self.open.pop_next_node())

        ret = self._select_random_node(nodes)

        for node in nodes:
            if node.state != ret.state:
                self.open.push_node(node)

        self.T *= self.T_scale_factor

        self.close.add_node(ret)
        return ret

