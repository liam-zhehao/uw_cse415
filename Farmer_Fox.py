'''Farmer_Fox.py
[STUDENTS: REPLACE THE FOLLOWING INFORMATION WITH YOUR
OWN:]
by Zhehao Li
UWNetIDs: zhehao
Student numbers: 2326829

Assignment 2, in CSE 415, Winter 2025
 
This file contains my problem formulation for the problem of
the Farmer, Fox, Chicken, and Grain.
'''

# Put your formulation of the Farmer-Fox-Chicken-and-Grain problem here.
# Be sure your name(s), uwnetid(s), and 7-digit student number(s) are given above in 
# the format shown.

# You should model your code closely after the given example problem
# formulation in HumansRobotsFerry.py

# Put your metadata here, in the same format as in HumansRobotsFerry.

# Start your Common Code section here.

'''Farmer_Fox.py
Implementation of Farmer, Fox, Chicken, and Grain Problem
'''
#<METADATA>
PROBLEM_NAME = "Farmer, Fox, Chicken, and Grain"
PROBLEM_VERSION = "1.0"
PROBLEM_AUTHORS = ['Not Sure']
PROBLEM_CREATION_DATE = "20-JAN-2025"
#</METADATA>

#<COMMON_CODE>
LEFT = 0
RIGHT = 1

class State:
    # include methods similar to those in HumansRobotsFerry.py for
    # this class.
    def __init__(self, old=None):
        if old is None:
            self.farmer = LEFT
            self.fox = LEFT
            self.chicken = LEFT
            self.grain = LEFT
        else:
            self.farmer = old.farmer
            self.fox = old.fox
            self.chicken = old.chicken
            self.grain = old.grain

    def __eq__(self, s2):
        if self.farmer != s2.farmer: return False
        if self.fox != s2.fox: return False
        if self.chicken != s2.chicken: return False
        if self.grain != s2.grain: return False
        return True

    def __str__(self):
        # show what's on each side
        left = []
        right = []
        for item in ['farmer', 'fox', 'chicken', 'grain']:
            if getattr(self, item) == LEFT:
                left.append(item)
            else:
                right.append(item)
        return f"Left bank: {', '.join(left)}\nRight bank: {', '.join(right)}\n"

    def __hash__(self):
        return (self.__str__()).__hash__()

    def copy(self):
        return State(old = self)

    def can_move(self, item):
        """ Determine whether it is valid for the farmer to move along with the specified item. """
        # Ensure the item is on the same side as the farmer
        if item and getattr(self, item) != self.farmer:
            return False

        # Simulate the state after the move
        new_state = self.move(item)

        # Check if the new state is safe
        # Left bank and right bank items
        left = []
        right = []
        for attr in ['farmer', 'fox', 'chicken', 'grain']:
            if getattr(new_state, attr) == LEFT:
                left.append(attr)
            else:
                right.append(attr)

        # Check if either bank has an unsafe situation
        if ('fox' in left and 'chicken' in left and 'farmer' not in left):
            return False  # Fox and chicken together without the farmer on the left bank
        if ('chicken' in left and 'grain' in left and 'farmer' not in left):
            return False  # Chicken and grain together without the farmer on the left bank
        if ('fox' in right and 'chicken' in right and 'farmer' not in right):
            return False  # Fox and chicken together without the farmer on the right bank
        if ('chicken' in right and 'grain' in right and 'farmer' not in right):
            return False  # Chicken and grain together without the farmer on the right bank
        return True

    def move(self, item):
        '''Assuming it's legal to make the move, this computes
        the new state resulting from moving the ferry carrying
        farmer, fox, chicken, grain.'''
        new_state = self.copy()
        new_state.farmer = 1 - new_state.farmer  # Farmer switches sides
        if item:
            setattr(new_state, item, 1 - getattr(new_state, item))  # Move the specified item
        return new_state

    def is_goal(self):
        return all(getattr(self, x) == RIGHT for x in ['farmer', 'fox', 'chicken', 'grain'])

# Finish off with the GOAL_TEST and GOAL_MESSAGE_FUNCTION here.

def goal_message(s):
    return "Congratulations on successfully guiding farmer, fox, chicken, grain across the river!"

GOAL_MESSAGE_FUNCTION = goal_message
def GOAL_TEST(state):
    return state.is_goal()

#<INITIAL_STATE>
CREATE_INITIAL_STATE = lambda : State()
#</INITIAL_STATE>

# Put your OPERATORS section here.
class Operator:
    def __init__(self, name, precond, state_transf):
        self.name = name
        self.precond = precond
        self.state_transf = state_transf

    def is_applicable(self, s):
        return self.precond(s)

    def apply(self, s):
        return self.state_transf(s)

OPERATORS = [Operator(f"Farmer crosses with {item}",
                      lambda s, i=item: s.can_move(i),
                      lambda s, i=item: s.move(i))
             for item in [None, 'fox', 'chicken', 'grain']]







