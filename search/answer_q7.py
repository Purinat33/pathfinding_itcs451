from typing import Literal, List, Tuple, TypeAlias, Annotated

import numpy as np
import queue

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
    x, y = np.where(grid >= 20)
    if len(x) > 0 and len(y) > 0:
        x, y = x[0], y[0]
        if grid[x][y] == 20:
            facing = 'N'
        elif grid[x][y] == 30:
            facing = 'E'
        elif grid[x][y] == 40:
            facing = 'S'
        elif grid[x][y] == 50:
            facing = 'W'
        return x, y, facing
    else:
        return None, None, None

ACTIONS: List[str] = ['Turn left', 'Turn right', 'Move forward']

def transition(state: State, action: str, grid: np.ndarray) -> State:
    """Return a new state."""
    x, y, facing = state
    new_facing = facing
    if action == 'Turn left':
        if facing == 'N':
            new_facing = 'W'
        elif facing == 'E':
            new_facing = 'N'
        elif facing == 'S':
            new_facing = 'E'
        elif facing == 'W':
            new_facing = 'S'
    elif action == 'Turn right':
        if facing == 'N':
            new_facing = 'E'
        elif facing == 'E':
            new_facing = 'S'
        elif facing == 'S':
            new_facing = 'W'
        elif facing == 'W':
            new_facing = 'N'
    elif action == 'Move forward':
        if facing == 'N':
            if x > 0 and grid[x - 1][y] != 1:
                x -= 1
        elif facing == 'E':
            if y < grid.shape[1] - 1 and grid[x][y + 1] != 1:
                y += 1
        elif facing == 'S':
            if x < grid.shape[0] - 1 and grid[x + 1][y] != 1:
                x += 1
        elif facing == 'W':
            if y > 0 and grid[x][y - 1] != 1:
                y -= 1
    return x, y, new_facing

def is_goal(state: State, grid: np.ndarray) -> bool:
    """Return whether the state is a goal state."""
    x, y, _ = state
    return grid[x][y] == 6

def cost(state: State, action: str, grid: np.ndarray) -> float:
    """Return a cost of an action on the state."""
    x, y, _ = state
    if action == 'Move forward':
        cell_value = grid[x][y]
        if cell_value == 0:
            return 1.0
        elif cell_value == 7:
            return 2.0
        elif cell_value == 8:
            return 3.0
    return 0.0

#===============================================================================
# 7.2 SEARCH
#===============================================================================

def heuristic(state: State, goal_state: State) -> float:
    """Return the heuristic value of the state."""
    x1, y1, _ = state
    x2, y2, _ = goal_state
    return abs(x1 - x2) + abs(y1 - y2)

def graph_search(
        grid: np.ndarray,
        strategy: Literal['DFS', 'BFS', 'UCS', 'GS', 'A*'] = 'A*'
        ) -> Tuple[
            Annotated[List[str], 'actions of the plan'],
            Annotated[List[State], 'states of the plan'],
            Annotated[List[State], 'explored states']]:
    """Return a plan (actions and states) and a list of explored states (in order)."""

    def reconstruct_path(came_from, current):
        path = []
        while current in came_from:
            action, current = came_from[current]
            path.append(action)
        path.reverse()
        return path

    start_state = state_func(grid)
    goal_state = (grid.shape[0] - 2, grid.shape[1] - 2, 'S')

    if strategy == 'DFS':
        frontier = [start_state]
    elif strategy == 'BFS':
        frontier = queue.Queue()
        frontier.put(start_state)
    elif strategy == 'UCS':
        frontier = queue.PriorityQueue()
        frontier.put((0, start_state))
        cost_so_far = {start_state: 0}
    elif strategy == 'GS':
        frontier = queue.PriorityQueue()
        frontier.put((heuristic(start_state, goal_state), start_state))
    elif strategy == 'A*':
        frontier = queue.PriorityQueue()
        frontier.put((0 + heuristic(start_state, goal_state), start_state))
        cost_so_far = {start_state: 0}

    came_from = {}
    explored_states = []

    while not frontier.empty():
        if strategy == 'DFS':
            current = frontier.pop()
        elif strategy == 'BFS':
            current = frontier.get()
        elif strategy == 'UCS' or strategy == 'GS' or strategy == 'A*':
            _, current = frontier.get()

        if current == goal_state:
            plan_actions = reconstruct_path(came_from, goal_state)
            plan_states = [start_state]
            for action in plan_actions:
                next_state = transition(plan_states[-1], action, grid)
                plan_states.append(next_state)
            return plan_actions, plan_states, explored_states

        explored_states.append(current)

        for action in ACTIONS:
            next_state = transition(current, action, grid)

            if next_state is None:
                continue

            new_cost = cost_so_far[current] + cost(next_state, action, grid)

            if strategy == 'UCS' or strategy == 'A*':
                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    priority = new_cost + heuristic(next_state, goal_state)
                    frontier.put((priority, next_state))
                    came_from[next_state] = (action, current)
            else:
                if next_state not in came_from and next_state not in explored_states:
                    frontier.append(next_state)
                    came_from[next_state] = (action, current)

    return [], [], []
