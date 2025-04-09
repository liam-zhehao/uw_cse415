"""
EightPuzzleWithManhattan.py
This file augments EightPuzzle.py with heuristic information,
so that it can be used by an A* implementation.
"""

from EightPuzzle import *

def h(state):
    """
    Manhattan distance heuristic:
    Computes the sum of the Manhattan distances of all tiles (excluding the blank tile).
    """
    goal_positions = {
        0: (0, 0), 1: (0, 1), 2: (0, 2),
        3: (1, 0), 4: (1, 1), 5: (1, 2),
        6: (2, 0), 7: (2, 1), 8: (2, 2)
    }
    distance = 0
    for i in range(3):
        for j in range(3):
            tile = state.b[i][j]
            if tile != 0:
                goal_i, goal_j = goal_positions[tile]
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance