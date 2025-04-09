"""
EightPuzzleWith.py
This file augments EightPuzzle.py with heuristic information,
so that it can be used by an A* implementation.
"""

from EightPuzzle import *

def h(state):
    """
    Compute Manhattan Distance first, and add extra distance.
    """
    goal_positions = {
        0: (0, 0), 1: (0, 1), 2: (0, 2),
        3: (1, 0), 4: (1, 1), 5: (1, 2),
        6: (2, 0), 7: (2, 1), 8: (2, 2)
    }
    distance = 0
    blank_pos = None

    # Compute Manhattan distance and record blank position
    for i in range(3):
        for j in range(3):
            tile = state.b[i][j]
            if tile != 0:
                goal_i, goal_j = goal_positions[tile]
                distance += abs(i - goal_i) + abs(j - goal_j)

            if state.b[i][j] == 0:
                blank_pos = (i, j)

    if blank_pos:
        blank_i, blank_j = blank_pos
        adjacent_positions = [
            (blank_i - 1, blank_j),  # Up
            (blank_i + 1, blank_j),  # Down
            (blank_i, blank_j - 1),  # Left
            (blank_i, blank_j + 1)   # Right
        ]
        """
        If the target position of a number lies in the direction of the blank tile (row or column), 
        an additional cost of 1 is added because the blank tile may require more moves 
        to get certain numbers reach their target positions.
        """
        for adj_i, adj_j in adjacent_positions:
            if 0 <= adj_i < 3 and 0 <= adj_j < 3:
                adjacent_tile = state.b[adj_i][adj_j]
                if adjacent_tile != 0:
                    goal_i, goal_j = goal_positions[adjacent_tile]
                    if (adj_i, adj_j) != (goal_i, goal_j):
                        if (goal_i == blank_i and abs(goal_j - blank_j) == 1) or \
                                (goal_j == blank_j and abs(goal_i - blank_i) == 1):
                            distance += 1

    return distance

"""
python AStar.py EightPuzzleWithMyImplement '[[3,0,1],[6,4,2],[7,8,5]]'
"""