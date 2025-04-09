'''TowersOfHanoi.py
'''
# <METADATA>
PROBLEM_NAME = "Towers of Hanoi"
PROBLEM_VERSION = "0.3"
PROBLEM_AUTHORS = ['S. Tanimoto']
PROBLEM_CREATION_DATE = "10-Jan-2025"
PROBLEM_DESC = \
    '''This formulation of the Towers of Hanoi problem uses generic
Python 3 constructs and has been tested with Python 3.12
'''
# </METADATA>

# <COMMON_DATA>
N_disks = 4  # Use default, but override if new value supplied
# by the user on the command line.
try:
    import sys

    arg2 = sys.argv[2]
    N_disks = int(arg2)
    print("Number of disks is " + arg2)
except:
    print("Using default number of disks: " + str(N_disks))
    print(" (To use a specific number, enter it on the command line, e.g.,")
    print("python3 ../Int_Solv_Client.py TowersOfHanoi 3")


# </COMMON_DATA>

# <COMMON_CODE>
class State:
    def __init__(self, old=None):
        if old is None:
            self.piles = [list(range(N_disks, 0, -1)), [], []]
        else:
            self.piles = [lst[:] for lst in old.piles]
        
    def __eq__(self, s2):
        for p in range(3):
            if self.piles[p] != old.piles[p]: return False
        return True

    def __str__(self):
        # Produces a textual description of a state.
        # Might not be needed in normal operation with GUIs.
        txt = "["
        for pile in self.piles:
            txt += str(pile) + " ,"
        return txt[:-2] + "]"

    def __hash__(self):
        return (self.__str__()).__hash__()

    def copy(self):
        # Performs an appropriately deep copy of a state,
        # for use by operators in creating new states.
        return State(old=self)

    def can_move(self, From, To):
        '''Tests whether it's legal to move a disk in state s
       from the From peg to the To peg.'''
        try:
            pf = self.piles[From]  # peg disk goes from
            pt = self.piles[To]  # peg disk goes to
            if pf == []: return False  # no disk to move.
            df = pf[-1]  # get topmost disk at From peg..
            if pt == []: return True  # no disk to worry about at To peg.
            dt = pt[-1]  # get topmost disk at To peg.
            if df < dt: return True  # Disk is smaller than one it goes on.
            return False  # Disk too big for one it goes on.
        except (Exception) as e:
            print(e)

    def move(self, From, To):
        '''Assuming it's legal to make the move, this computes
       the new state resulting from moving the topmost disk
       from the From peg to the To peg.'''
        news = self.copy()  # start with a deep copy.
        pf = self.piles[From]  # peg disk goes from.
        pt = self.piles[To]
        df = pf[-1]  # the disk to move.
        news.piles[From] = pf[:-1]  # remove it from its old peg.
        news.piles[To] = pt[:] + [df]  # Put disk onto destination peg.
        return news  # return new state

    def is_goal(self):
       '''If the first two pegs are empty, then s is a goal state.'''
       return self.piles[0] == [] and self.piles[1] == []


def goal_message(s):
    return "The Tower Transport is Triumphant!"

class Operator:
    def __init__(self, name, precond, state_transf):
        self.name = name
        self.precond = precond
        self.state_transf = state_transf

    def is_applicable(self, s):
        return self.precond(s)

    def apply(self, s):
        return self.state_transf(s)


# </COMMON_CODE>

# <INITIAL_STATE>
CREATE_INITIAL_STATE = lambda: State()
# </INITIAL_STATE>

# <OPERATORS>
peg_combinations = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
OPERATORS = [Operator("Move disk from " + str(p) + " to " + str(q),
                      lambda s, p1=p-1, q1=q-1: s.can_move(p1, q1),
                      # The default value construct is needed
                      # here to capture the values of p&q separately
                      # in each iteration of the list comp. iteration.
                      lambda s, p1=p-1, q1=q-1: s.move(p1, q1))
             for (p, q) in peg_combinations]
# </OPERATORS>

# <GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
# </GOAL_MESSAGE_FUNCTION>
