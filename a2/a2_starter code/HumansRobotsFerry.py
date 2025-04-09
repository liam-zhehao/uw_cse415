'''HumansRobotsFerry.py
("Humans, Robots and Ferry" problem)
'''
#<METADATA>
PROBLEM_NAME = "Humans, Robots, and Ferry"
PROBLEM_VERSION = "1.1"
PROBLEM_AUTHORS = ['S. Tanimoto']
PROBLEM_CREATION_DATE = "10-JAN-2025"

# The following field is mainly for the human solver, via either the Text_SOLUZION_Client.
# or the SVG graphics client.
PROBLEM_DESC=\
 '''The <b>"Humans, Robots and Ferry"</b> problem is a variation of
the classic puzzle "Missionaries and Cannibals." In the Humans, Robots
and Ferry problem, the player starts off with three humans and three
robots on the left bank of a creek.  The object is to execute a
sequence of legal moves that transfers them all to the right bank of
the creek.  In this puzzle, there is a ferry that can carry at most
three agents (humans, robots), and one of them must be a human to steer
the ferry.  It is forbidden to ever have one or two humans outnumbered
by robots, either on the left bank, right bank, or on the ferry.
In the formulation presented here, the computer will not let you make a
move to such a forbidden situation, and it will only show you moves
that could be executed "safely."
'''
#</METADATA>

#<COMMON_DATA>
#</COMMON_DATA>

#<COMMON_CODE>
Humans_on_left = 3
Robots_on_left = 3
LEFT=0 # same idea for left side of creek
RIGHT=1 # etc.

class State():

  def __init__(self, old=None):
    if old is None: 
      self.humans_on_left = 3
      self.robots_on_left = 3
      self.ferry=LEFT
    else:
      self.humans_on_left = old.humans_on_left
      self.robots_on_left = old.robots_on_left
      self.ferry = old.ferry

  def __eq__(self,s2):
    if self.ferry != s2.ferry: return False
    if self.humans_on_left != s2.humans_on_left: return False    
    if self.robots_on_left != s2.robots_on_left: return False    
    return True

  def __str__(self):
    # Produces a textual description of a state.
    txt = "\n H on left:"+str(self.humans_on_left)+"\n"
    txt += " R on left:"+str(self.robots_on_left)+"\n"
    txt += "   H on right:"+str(3 - self.humans_on_left)+"\n"
    txt += "   R on right:"+str(3 - self.robots_on_left)+"\n"
    if self.ferry == LEFT:
      txt += " ferry is on the left.\n"
    else:
      txt += " ferry is on the right.\n"
    return txt

  def __hash__(self):
    return (self.__str__()).__hash__()

  def copy(self):
    # Performs an appropriately deep copy of a state,
    # for use by operators in creating new states.
    return State(old=self)

  def can_move(self,h,r):
    '''Tests whether it's legal to move the ferry and take
     h humans and r robots.'''
    side = self.ferry # Where the ferry is.
    if h<1: return False # Need an H to steer boat.
    h_available = self.humans_on_left if side==LEFT else 3-self.humans_on_left
    if h_available < h: return False # Can't take more h's than available
    r_available = self.robots_on_left if side==LEFT else 3-self.robots_on_left
    if r_available < r: return False # Can't take more r's than available
    h_remaining = h_available - h
    r_remaining = r_available - r
    # Humans must not be outnumbered on either side:
    if h_remaining > 0 and h_remaining < r_remaining: return False
    h_at_arrival = 3 - self.humans_on_left + h if side==LEFT else self.humans_on_left + h
    r_at_arrival = 3 - self.robots_on_left + r if side==LEFT else self.robots_on_left + r
    if h_at_arrival > 0 and h_at_arrival < r_at_arrival: return False
    return True


  def move(self,h,r):
    '''Assuming it's legal to make the move, this computes
     the new state resulting from moving the ferry carrying
     h humans and r robots.'''
    news = self.copy()      # start with a deep copy.
    # Remove agents from the current side.
    if self.ferry == LEFT: 
      news.humans_on_left -= h
      news.robots_on_left -= r
    else: 
      news.humans_on_left += h
      news.robots_on_left += r
    news.ferry = 1 - self.ferry
    return news

  def is_goal(self):
    '''If all Hs and Rs are on the right, then this is a goal state.'''
    if self.humans_on_left==0 and self.robots_on_left==0: return True
    else: return False

def goal_message(s):
  return "Congratulations on successfully guiding the humans and robots across the creek!"

class Operator:
  def __init__(self, name, precond, state_transf):
    self.name = name
    self.precond = precond
    self.state_transf = state_transf

  def is_applicable(self, s):
    return self.precond(s)

  def apply(self, s):
    return self.state_transf(s)
#</COMMON_CODE>

#<INITIAL_STATE>
CREATE_INITIAL_STATE = lambda : State()
#</INITIAL_STATE>

#<OPERATORS>
HR_combinations = [(1,0),(2,0),(3,0),(1,1),(2,1)]

OPERATORS = [Operator(
  "Cross the creek with "+str(h)+" humans and "+str(r)+" robots",
  lambda s, h1=h, r1=r: s.can_move(h1,r1),
  lambda s, h1=h, r1=r: s.move(h1,r1) ) 
  for (h,r) in HR_combinations]
#</OPERATORS>

