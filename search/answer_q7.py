import numpy as np
import heapq
from typing import List, Tuple, TypeAlias, Annotated, Literal

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
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            if grid[x][y] in [20, 21, 22]:
                return x, y, 'N'
            elif grid[x][y] in [30, 31, 32]:
                return x, y, 'E'
            elif grid[x][y] in [40, 41, 42]:
                return x, y, 'S'
            elif grid[x][y] in [50, 51, 52]:
                return x, y, 'W'
    return None

ACTIONS: List[str] = ['Turn left', 'Turn right', 'Move forward']

def transition(state: State, action: str, grid: np.ndarray) -> State:
    """Return a new state."""
    x, y, facing = state
    orientation = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
    
    if action == 'Turn left':
        new_facing = ['N', 'E', 'S', 'W'][(orientation[facing] - 1) % 4]
        return x, y, new_facing

    if action == 'Turn right':
        new_facing = ['N', 'E', 'S', 'W'][(orientation[facing] + 1) % 4]
        return x, y, new_facing

    if action == 'Move forward':
        if facing == 'N':
            new_x, new_y = x, y - 1
        elif facing == 'E':
            new_x, new_y = x + 1, y
        elif facing == 'S':
            new_x, new_y = x, y + 1
        else:  # facing == 'W'
            new_x, new_y = x - 1, y

        if grid[new_x][new_y] not in [1]:  # Can't move through walls
            return new_x, new_y, facing

    return x, y, facing  # If action is not recognized, stay in the same state

def is_goal(state: State, grid: np.ndarray) -> bool:
    """Return whether the state is a goal state."""
    x, y, _ = state
    return grid[x][y] == 6

def cost(state: State, action: str, grid: np.ndarray) -> float:
    """Return a cost of an action on the state."""
    x, y, _ = state
    if grid[x][y] == 7:
        return 1.5  # Cost of moving through mud
    elif grid[x][y] == 8:
        return 1.2  # Cost of moving through grass
    else:
        return 1.0  # Default cost

#===============================================================================
# 7.2 SEARCH
#===============================================================================

def heuristic(state: State, goal_state: State) -> float:
    """Return the heuristic value of the state."""
    x1, y1, _ = state
    x2, y2, _ = goal_state
    return abs(x1 - x2) + abs(y1 - y2)

# def graph_search(
#         grid: np.ndarray,
#         strategy: Literal['DFS', 'BFS', 'UCS', 'GS', 'A*'] = 'A*'
#         ) -> Tuple[
#             Annotated[List[str], 'actions of the plan'],
#             Annotated[List[State], 'states of the plan'],
#             Annotated[List[State], 'explored states']]:
#     """Return a plan (actions and states) and a list of explored states (in order)."""

#     start_state = state_func(grid)
#     goal_state = (len(grid) - 2, len(grid) - 2, 'E')  # Goal is always at the bottom-right corner facing East

#     explored_states = []
#     plan_states = []
#     plan_actions = []

#     # Priority queue for A* using heapq
#     open_list = [(0, start_state)]  # (f-cost, state)
#     came_from = {}  # Dictionary to store the parent state for each state
#     g_costs = {start_state: 0}  # Dictionary to store g-cost for each state

#     while open_list:
#         _, current_state = heapq.heappop(open_list)  # Get the state with the lowest f-cost
#         explored_states.append(current_state)

#         if current_state == goal_state:
#             # Reconstruct the path
#             while current_state in came_from:
#                 plan_states.insert(0, current_state)
#                 current_state = came_from[current_state]
#             break

#         for action in ACTIONS:
#             new_state = transition(current_state, action, grid)
#             if new_state:
#                 tentative_g = g_costs[current_state] + cost(current_state, action, grid)

#                 if new_state not in g_costs or tentative_g < g_costs[new_state]:
#                     g_costs[new_state] = tentative_g
#                     f_cost = tentative_g + heuristic(new_state, goal_state)
#                     heapq.heappush(open_list, (f_cost, new_state))
#                     came_from[new_state] = current_state

#     # Extract plan actions from plan states
#     for i in range(len(plan_states) - 1):
#         current_state = plan_states[i]
#         next_state = plan_states[i + 1]
#         dx = next_state[0] - current_state[0]
#         dy = next_state[1] - current_state[1]

#         if dx == 1:
#             plan_actions.append('Move forward')
#         elif dx == -1:
#             plan_actions.append('Turn right')
#             plan_actions.append('Move forward')
#         elif dy == 1:
#             plan_actions.append('Turn left')
#             plan_actions.append('Move forward')
#         elif dy == -1:
#             plan_actions.append('Turn right')
#             plan_actions.append('Turn right')
#             plan_actions.append('Move forward')

#     return plan_actions, plan_states, explored_states

def dfs(grid, start_state, goal_state):
    stack = [start_state]
    came_from = {}
    explored_states = []

    while stack:
        current_state = stack.pop()
        explored_states.append(current_state)

        if current_state == goal_state:
            # Reconstruct the path
            plan_states = []
            plan_actions = []
            while current_state in came_from:
                plan_states.insert(0, current_state)
                current_state = came_from[current_state]
            return plan_actions, plan_states, explored_states

        for action in ACTIONS:
            new_state = transition(current_state, action, grid)
            if new_state and new_state not in came_from:
                came_from[new_state] = current_state
                stack.append(new_state)

    return [], [], explored_states

def bfs(grid, start_state, goal_state):
    queue = [start_state]
    came_from = {}
    explored_states = []

    while queue:
        current_state = queue.pop(0)
        explored_states.append(current_state)

        if current_state == goal_state:
            # Reconstruct the path
            plan_states = []
            plan_actions = []
            while current_state in came_from:
                plan_states.insert(0, current_state)
                current_state = came_from[current_state]
            return plan_actions, plan_states, explored_states

        for action in ACTIONS:
            new_state = transition(current_state, action, grid)
            if new_state and new_state not in came_from:
                came_from[new_state] = current_state
                queue.append(new_state)

    return [], [], explored_states

def ucs(grid, start_state, goal_state):
    priority_queue = [(0, start_state)]  # (g-cost, state)
    came_from = {}
    g_costs = {start_state: 0}
    explored_states = []

    while priority_queue:
        _, current_state = heapq.heappop(priority_queue)
        explored_states.append(current_state)

        if current_state == goal_state:
            # Reconstruct the path
            plan_states = []
            plan_actions = []
            while current_state in came_from:
                plan_states.insert(0, current_state)
                current_state = came_from[current_state]
            return plan_actions, plan_states, explored_states

        for action in ACTIONS:
            new_state = transition(current_state, action, grid)
            if new_state:
                tentative_g = g_costs[current_state] + cost(current_state, action, grid)

                if new_state not in g_costs or tentative_g < g_costs[new_state]:
                    g_costs[new_state] = tentative_g
                    heapq.heappush(priority_queue, (tentative_g, new_state))
                    came_from[new_state] = current_state

    return [], [], explored_states

def gs(grid, start_state, goal_state):
    priority_queue = [(0, start_state)]  # (heuristic, state)
    came_from = {}
    explored_states = []

    while priority_queue:
        _, current_state = heapq.heappop(priority_queue)
        explored_states.append(current_state)

        if current_state == goal_state:
            # Reconstruct the path
            plan_states = []
            plan_actions = []
            while current_state in came_from:
                plan_states.insert(0, current_state)
                current_state = came_from[current_state]
            return plan_actions, plan_states, explored_states

        for action in ACTIONS:
            new_state = transition(current_state, action, grid)
            if new_state:
                heuristic_value = heuristic(new_state, goal_state)
                heapq.heappush(priority_queue, (heuristic_value, new_state))
                came_from[new_state] = current_state

    return [], [], explored_states

def a_star(grid, start_state, goal_state):
    priority_queue = [(0, start_state)]  # (f-cost, state)
    came_from = {}
    g_costs = {start_state: 0}
    explored_states = []

    while priority_queue:
        _, current_state = heapq.heappop(priority_queue)
        explored_states.append(current_state)

        if current_state == goal_state:
            # Reconstruct the path
            plan_states = []
            plan_actions = []
            while current_state in came_from:
                plan_states.insert(0, current_state)
                current_state = came_from[current_state]
            return plan_actions, plan_states, explored_states

        for action in ACTIONS:
            new_state = transition(current_state, action, grid)
            if new_state:
                tentative_g = g_costs[current_state] + cost(current_state, action, grid)

                if new_state not in g_costs or tentative_g < g_costs[new_state]:
                    g_costs[new_state] = tentative_g
                    f_cost = tentative_g + heuristic(new_state, goal_state)
                    heapq.heappush(priority_queue, (f_cost, new_state))
                    came_from[new_state] = current_state

    return [], [], explored_states

def graph_search(
        grid: np.ndarray,
        strategy: Literal['DFS', 'BFS', 'UCS', 'GS', 'A*'] = 'A*'
        ) -> Tuple[
            Annotated[List[str], 'actions of the plan'],
            Annotated[List[State], 'states of the plan'],
            Annotated[List[State], 'explored states']]:
    """Return a plan (actions and states) and a list of explored states (in order)."""

    start_state = state_func(grid)
    goal_state = (len(grid) - 2, len(grid) - 2, 'E')  # Goal is always at the bottom-right corner facing East

    if strategy == 'DFS':
        return dfs(grid, start_state, goal_state)
    elif strategy == 'BFS':
        return bfs(grid, start_state, goal_state)
    elif strategy == 'UCS':
        return ucs(grid, start_state, goal_state)
    elif strategy == 'GS':
        return gs(grid, start_state, goal_state)
    elif strategy == 'A*':
        return a_star(grid, start_state, goal_state)
    else:
        raise ValueError("Invalid strategy")

# Example usage:
# plan_actions, plan_states, explored_states = graph_search(grid, strategy='DFS')
# or
# plan_actions, plan_states, explored_states = graph_search(grid, strategy='BFS')
# ... and so on for other strategies
