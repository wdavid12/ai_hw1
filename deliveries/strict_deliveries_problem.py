from framework.graph_search import *
from framework.ways import *
from .map_problem import MapProblem
from .deliveries_problem_input import DeliveriesProblemInput
from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem

from typing import Set, FrozenSet, Optional, Iterator, Tuple, Union


class StrictDeliveriesState(RelaxedDeliveriesState):
    """
    An instance of this class represents a state of the strict
     deliveries problem.
    This state is basically similar to the state of the relaxed
     problem. Hence, this class inherits from `RelaxedDeliveriesState`.

    TODO:
        If you believe you need to modify the state for the strict
         problem in some sense, please go ahead and do so.
    """
    def __init__(self, current_location: Junction,
                 dropped_so_far: Union[Set[Junction], FrozenSet[Junction]],
                 fuel: float, problem_input: DeliveriesProblemInput ):
        super().__init__(current_location, dropped_so_far, fuel)
        self.problem_input = problem_input


class StrictDeliveriesProblem(RelaxedDeliveriesProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'StrictDeliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 inner_problem_solver: GraphProblemSolver, use_cache: bool = True):
        super(StrictDeliveriesProblem, self).__init__(problem_input)
        self.initial_state = StrictDeliveriesState(
            problem_input.start_point, frozenset(), problem_input.gas_tank_init_fuel, problem_input)
        self.problem_input = problem_input
        self.inner_problem_solver = inner_problem_solver
        self.roads = roads
        self.use_cache = use_cache
        self._init_cache()

    def _init_cache(self):
        self._cache = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key, val):
        if self.use_cache:
            self._cache[key] = val

    def _get_from_cache(self, key):
        if not self.use_cache:
            return None
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return self._cache.get(key)

    def _calc_distance(self, source: Junction, dest: Junction) -> float:
        pair = frozenset({source,dest})
        known_distance = self._get_from_cache(pair)
        if known_distance is not None:
            return known_distance

        problem = MapProblem(self.roads, source.index, dest.index)
        distance = self.inner_problem_solver.solve_problem(problem).final_search_node.cost

        self._insert_to_cache(pair, distance)

        return distance

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        """
        TODO: implement this method!
        This method represents the `Succ: S -> P(S)` function of the strict deliveries problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The relaxed problem operators are defined in the assignment instructions.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, a pair of the successor state and the operator cost is yielded.
        """
        assert isinstance(state_to_expand, StrictDeliveriesState)

        remaining = self.drop_points - state_to_expand.dropped_so_far

        for junction in remaining:
            distance = self._calc_distance(state_to_expand.current_location, junction)
            if state_to_expand.fuel < distance:
                continue
            remaining_fuel = state_to_expand.fuel - distance
            new_dropped_so_far = state_to_expand.dropped_so_far | frozenset([junction])
            yield StrictDeliveriesState(junction, new_dropped_so_far, remaining_fuel, self.problem_input), distance

        for station in self.gas_stations:
            distance = self._calc_distance(state_to_expand.current_location, station)
            if state_to_expand.fuel < distance:
                continue
            yield StrictDeliveriesState(station, state_to_expand.dropped_so_far, self.gas_tank_capacity, self.problem_input), distance


    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO: implement this method!
        """
        assert isinstance(state, StrictDeliveriesState)
        return super().is_goal(state)
