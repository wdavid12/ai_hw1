from .graph_problem_interface import *
from .best_first_search import BestFirstSearch


class AStar(BestFirstSearch):
    """
    This class implements the Weighted-A* search algorithm.
    A* algorithm is in the Best First Search algorithms family.
    """

    solver_name = 'A*'

    def __init__(self, heuristic_function_type: HeuristicFunctionType, heuristic_weight: float = 0.5):
        """
        :param heuristic_function_type: The A* solver stores the constructor of the heuristic
                                        function, rather than an instance of that heuristic.
                                        In each call to "solve_problem" a heuristic instance
                                        is created.
        :param heuristic_weight: Used to calculate the f-score of a node using
                                 the heuristic value and the node's cost. Default is 0.5.
        """
        # A* is a graph search algorithm. Hence, we use close set.
        super(AStar, self).__init__(use_close=True)
        self.heuristic_function_type = heuristic_function_type
        self.heuristic_weight = heuristic_weight
        self.solver_name += ' (h={heuristic_name}, w={heuristic_weight:.3f})'.format(
            heuristic_name=heuristic_function_type.heuristic_name,
            heuristic_weight=self.heuristic_weight)

    def _init_solver(self, problem):
        """
        Called by "solve_problem()" in the implementation of `BestFirstSearch`.
        The problem to solve is known now, so we can create the heuristic function to be used.
        """
        super(AStar, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        Called by solve_problem() in the implementation of `BestFirstSearch`
         whenever just after creating a new successor node.
        Should calculate and return the f-score of the given node.
        This score is used as a priority of this node in the open priority queue.

        TODO: implement this method.
        Remember: In Weighted-A* the f-score is defined by ((1-w) * cost) + (w * h(state)).
        Notice: You may use `search_node.cost`, `self.heuristic_weight`, and `self.heuristic_function`.
        """

        heuristic = self.heuristic_function.estimate(search_node.state)
        cost = search_node.cost

        return self.heuristic_weight*heuristic + (1-self.heuristic_weight)*cost


    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        """
        Called by solve_problem() in the implementation of `BestFirstSearch`
         whenever creating a new successor node.
        This method is responsible for adding this just-created successor
         node into the `self.open` priority queue, and may check the existence
         of another node representing the same state in `self.close`.
        """

        if self.open.has_state(successor_node.state):
            existing_node = self.open.get_node_by_state(successor_node.state)
            if existing_node.expanding_priority > successor_node.expanding_priority:
                self.open.extract_node(existing_node)
                self.open.push_node(successor_node)
        elif self.close.has_state(successor_node.state):
            existing_node = self.close.get_node_by_state(successor_node.state)
            if existing_node.expanding_priority > successor_node.expanding_priority:
                self.close.remove_node(existing_node)
                self.open.push_node(successor_node)
        else:
            self.open.push_node(successor_node)

