'''Special static evaluator for a specific set of
Tic-Tac-Toe states is provided here for testing
minimax and alpha-beta search.

'''

import game_types
SE_TABLE = {
'OOXXOX  X': 91,
'OOXXO X X': 10,
'OOXXO  XX': 20,
'OOXXXO  X': 11,
'OOXX OX X': 19,
'OOXX O XX': 11,
'OOXXX O X': 20,
'OOXX XO X': 109,
'OOXX  OXX': 11,
'OOXXX  OX': 30,
'OOXX X OX': 101,
'OOXX  XOX': 11 }

STATIC_EVAL_COUNT = 0
def special_static_eval_fn(state):
    global STATIC_EVAL_COUNT
    try: 
        code = state_quick_code(state)
    except:
        print("Warning: special static evaluation function is intended for TTT states only.")
        code = ''
    try:
        static_val = SE_TABLE[code]
    except KeyError:
        static_val = 0
        print("Warning: a state given to the special_static_eval_fn has no pre-computed static value in the table.")
    STATIC_EVAL_COUNT += 1
    return static_val

def state_quick_code(state):
    # Use as a hash, but there should never be collisions.
    b = state.board
    flat=sum(b, [])
    code="".join(flat)
    return code
