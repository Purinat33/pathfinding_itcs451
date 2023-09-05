from typing import Literal, List, Tuple, TypeAlias, Annotated

import numpy as np

State: TypeAlias = Tuple[int, int, str]

#===============================================================================
# 7.1 FORMULATION
#===============================================================================

def state_func(grid: np.ndarray) -> State:
    """Return a state based on the grid (observation).

    Number mapping:
    -  0: dirt (passable)
    -  1: wall (not passable)
    -  2x: agent is facing up (north)
    -  3x: agent is facing right (east)
    -  4x: agent is facing down (south)
    -  5x: agent is facing left (west)
    -  6: goal
    -  7: mud (passable, but cost more)
    -  8: grass (passable, but cost more)

    State is a tuple of
    - x (int)
    - y (int)
    - facing ('N', 'E', 'S', or 'W')
    """
    # TODO
    pass

# TODO
ACTIONS: List[str] = []

def transition(state: State, actoin: str, grid: np.ndarray) -> State:
    """Return a new state."""
    # TODO
    pass


def is_goal(state: State, grid: np.ndarray) -> bool:
    """Return whether the state is a goal state."""
    # TODO
    pass


def cost(state: State, actoin: str, grid: np.ndarray) -> float:
    """Return a cost of an action on the state."""
    # TODO
    # Place the following lines with your own implementation
    return 1.0


#===============================================================================
# 7.2 SEARCH
#===============================================================================


def heuristic(state: State, goal_state: State) -> float:
    """Return the heuristic value of the state."""
    # TODO
    pass


def graph_search(
        grid: np.ndarray,
        strategy: Literal['DFS', 'BFS', 'UCS', 'GS', 'A*'] = 'A*'
        ) -> Tuple[
            Annotated[List[str], 'actions of the plan'],
            Annotated[List[State], 'states of the plan'],
            Annotated[List[State], 'explored states']]:
    """Return a plan (actions and states) and a list of explored states (in order)."""

    # TODO
    # Replace the lines below with your own implementation
    init_state = (1, 1, 'W')
    return (
        ['www', 'xxx', 'yyy', 'zzz'],
        [(1, 1, 'W'), (1, 1, 'E'), (2, 1, 'E'), (2, 1, 'S')],
        [(1, 1, 'W'), (1, 1, 'E'), (1, 1, 'S'), (1, 2, 'S'), (1, 3, 'S'), (2, 1, 'E'), (2, 1, 'S')]
    )
