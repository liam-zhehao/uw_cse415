"""
EightPuzzleWithHamming.py
This file augments EightPuzzle.py with heuristic information,
so that it can be used by an A* implementation.
"""

from EightPuzzle import *

def h(state):
    """
    Hamming distance heuristic:
    Counts the number of misplaced tiles (excluding the blank tile).
    """
    goal = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    distance = 0
    for i in range(3):
        for j in range(3):
            if state.b[i][j] != 0 and state.b[i][j] != goal[i][j]:
                distance += 1
    return distance
