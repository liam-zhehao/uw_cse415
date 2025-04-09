''' autograder.py
A simple tester for K-in-a-Row agents. Intended as a debugging aid.

We assume the program being tested here ALREADY CAN PLAY
ALL TYPES OF K-IN-A-ROW GAMES WITH THE Game_Master_Offline.py
program.

Therefore, the tests here will be additional tests for
specific aspects of the agent's capabilities.

Version of Feb. 5, 2025

 tests:

1. Static eval  (worth 10 points)
2. Minimax
  2a. Straight minimax (worth 10 points)
  2b. Alpha-beta (worth 10 points)

Passing these tests does NOT mean that the agent is
necessarily complying with all assignment requirements.

Also, eligibility for the class tournament may be
determined using additional factors, such as what libaries
might be imported, timing behavior, etc.

To use this, change "YourUWNetID_KInARow to your actual agent file's
name, make sure you have the imported files in the same folder.
(These include both the ones your agent imports and the ones imported
by this autograder.)
Then run the command:

python autograder.py
  or on some systems:
python3 autograder.py

'''

import game_types
import YourUWNetID_KInARow as agent_module
import spec_static_by_table


def test_static_eval(agent):
    '''A static evaluation function should return high values
    on states that are good for X, and similarly strong but
    negative values on states that are good for O.
    Furthermore values should roughly track HOW good a
    state is for either player, and a state that's equally
    good for X and O should return a value of 0 or close to 0.

    The function should be robust to variations in the game
    type (i.e., initial state) and be able to work OK on
    whatever game states are presented to it, e.g., Tic-Tac-Toe,
    Cassini, etc. 

    Part 1: Tic-Tac-Toe
      State A:  win for O
      State B:  good for O
      State C:  neutral
      State D:  good for X
      State E:  win for X

    Part 2: Cassini
      State F:  win for O
      State G:  good for O
      State H:  neutral
      State I:  good for X
      State J:  win for X

'''

    Part_1_scores = [agent.static_eval(s, game_types.TTT) for s in [A, B, C, D, E]]
    num_inversions1 = count_inversions(Part_1_scores)

    #Part_2_scores = [0,0,0,0,0]
    Part_2_scores = [agent.static_eval(s, game_types.Cassini) for s in [F, G, H, I, J]]
    num_inversions2 = count_inversions(Part_2_scores)

    meta_score = (20 - (num_inversions1 + num_inversions2)) / 2
    #print("Your static_eval function gets ", meta_score, "out of 10.")
    return meta_score

def count_inversions(lst):
    n = len(lst)
    if n < 2: return 0

    # "Naive method" using nested loops; O(n^2) but code is simple
    # and we will have n = 5.  (Alternative uses modified Merge Sort and
    # needs 3 times as much code to achieve O(n log2 n) complexity.
    c = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if lst[i] >= lst[j]: c += 1
    return c

A = game_types.State(initial_state_data = \
    [[['X',' ','O'],
      [' ','O','X'],
      ['O',' ','X']], "X"])

B = game_types.State(initial_state_data = \
    [[['X',' ','O'],
      ['X',' ',' '],
      ['O','X',' ']], "O"])

C = game_types.State(initial_state_data = \
    [[['X',' ','O'],
      [' ',' ',' '],
      [' ',' ',' ']], "X"])

D = game_types.State(initial_state_data = \
    [[['O','X','X'],
      [' ',' ','O'],
      ['X','O',' ']], "X"])

E = game_types.State(initial_state_data = \
    [[['X','O','O'],
      ['X',' ','X'],
      ['X',' ','O']], "O"])

F = game_types.State(initial_state_data = \
    [[[' ',' ',' ',' ',' ',' ',' ','X'],
      [' ','O','O','O','O','O',' ',' '],
      [' ',' ',' ','-','-',' ',' ',' '],
      [' ',' ','-','-','-','-','X','X'],
      [' ',' ',' ','-','-',' ',' ',' '],
      [' ','X','X',' ',' ',' ',' ',' '],
      [' ',' ',' ',' ',' ',' ','O',' ']], "X"])

G = game_types.State(initial_state_data = \
    [[[' ',' ',' ',' ',' ',' ',' ','X'],
      [' ','O','O',' ','O','O',' ',' '],
      [' ','O',' ','-','-',' ',' ',' '],
      [' ',' ','-','-','-','-','X','X'],
      [' ',' ',' ','-','-',' ',' ',' '],
      [' ','X','X',' ',' ',' ',' ',' '],
      [' ',' ',' ',' ',' ','O','O',' ']], "X"])

H = game_types.State(initial_state_data = \
    [[[' ',' ',' ',' ',' ',' ',' ',' '],
      [' ','O',' ',' ',' ',' ','O','X'],
      [' ',' ',' ','-','-',' ',' ',' '],
      [' ',' ','-','-','-','-',' ',' '],
      [' ',' ',' ','-','-',' ',' ',' '],
      [' ','X',' ',' ',' ',' ','X',' '],
      ['O',' ',' ',' ',' ',' ','O',' ']], "O"])

I = game_types.State(initial_state_data = \
    [[[' ',' ',' ','O',' ',' ',' ','X'],
      [' ','O',' ',' ',' ','O','O','X'],
      ['O',' ',' ','-','-','O',' ','X'],
      [' ',' ','-','-','-','-','X',' '],
      [' ',' ',' ','-','-','X',' ',' '],
      [' ','X','X',' ',' ',' ',' ',' '],
      [' ',' ','O',' ',' ',' ','O',' ']], "X"])

J = game_types.State(initial_state_data = \
    [[[' ','O',' ',' ',' ',' ',' ','O'],
      [' ','O',' ',' ',' ',' ',' ','O'],
      [' ','O',' ','-','-',' ',' ',' '],
      [' ',' ','-','-','-','-','X','O'],
      [' ',' ',' ','-','-',' ',' ',' '],
      [' ','X','X','X','X','X',' ',' '],
      [' ',' ',' ',' ',' ',' ','O',' ']], "O"])

MM_TEST_STATE = game_types.State(initial_state_data = \
    [[['O','O','X'],
      ['X',' ',' '],
      [' ',' ','X']], "O"])
                                
def test_minimax(agent, use_alpha_beta=False):
    '''Autograde the minimax capability of the agent as follows.
    Call the agent's prepare method, in case it needs to
    do any initialization prior to making moves.
    Call its make_move function with:
      autograding=True
      use_alpha_beta=Fase
      special_static_eval_fn = TTT_static_eval.
      This method will count the number of times it has been
      called and store it in NUM_CALLS_TO_STATIC_EVAL.
      It should also return best move and new state;
      hopefully the backed-up value of the new state, too.
'''
    spec_static_by_table.STATIC_EVAL_COUNT = 0
    GAME_TYPE = game_types.TTT
    prep_response = agent.prepare(GAME_TYPE, 'X', "Op Ponent")
    if prep_response != 'OK':
        print("Agent did not respond properly to the prepare command.")
        return
    print("Prepare was successful, it seems.")
    print("Next ... let's test minimaxing (and possibly alpha-beta pruning).")
    opponent_remark = "Do your thing."
    response = agent.make_move(\
                MM_TEST_STATE,
                opponent_remark,
                autograding=True,
                use_alpha_beta=use_alpha_beta,
                use_zobrist_hashing=False,
                max_ply=2,
                special_static_eval_fn=spec_static_by_table.special_static_eval_fn)

    print("Agent ", agent.nickname, " gives the following make_move response:")
    print(response)
    se_count = spec_static_by_table.STATIC_EVAL_COUNT
    print(se_count, "calls made to the special static evaluation function.")
    best_move = response[0][0]
    if best_move==(1,2):
        meta_minimax_score = 5
    else:
        print("minimax did NOT find the correct move")
        meta_minimax_score = 0
    if use_alpha_beta:
        if se_count == 8:
            meta_minimax_score += 5
        else:
            print(se_count, "is the wrong number of leaves evaluated during alpha-beta.")
            print("The correct number is 8.")
    
    if not use_alpha_beta and se_count == 12:
        meta_minimax_score += 5
    else:
        print("Without alpha-beta, your search should evaluate 12 leaf nodes.")
        print("Your search did", se_count)
    return meta_minimax_score        
    
if __name__ == "__main__":

    agent = agent_module.OurAgent()
    meta_se_score = test_static_eval(agent)
    print("The static_evaluation function's meta-score for ",
          agent.nickname, " is", meta_se_score)

    meta_minimax_score = test_minimax(agent)
    if meta_minimax_score==10:
        print("Your agent PASSES the minimax test!")
    meta_alpha_beta_score = test_minimax(agent, use_alpha_beta=True)
    if meta_alpha_beta_score==10:
        print("Your agent PASSES the alpha-beta test!")
    total_autograder_score = meta_se_score + meta_minimax_score + meta_alpha_beta_score    
    print("Your agent's total autograded score is ", total_autograder_score, "out of 30.0.") 
