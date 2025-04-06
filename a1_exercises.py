# Zhehao Li 2326829
# CSE 415, Assignment 1, Winter 2025.
import copy
import math
import re


def is_a_quintuple(n):
    """Return True if n is a multiple of 5; False otherwise."""
    return n % 5 == 0

def last_prime(m):
    """Return the largest prime number p that is less than or equal to m.
    You might wish to define a helper function for this.
    You may assume m is a positive integer."""

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    for p in range(m, 1, -1):
        if is_prime(p):
            return p
    return None

def quadratic_roots(a, b, c):
    """Return the roots of a quadratic equation (real cases only).
    Return results in tuple-of-floats form, e.g., (-7.0, 3.0)
    Return "complex" if real roots do not exist."""
    d = b ** 2 - 4 * a * c
    if d < 0:
        return "complex"
    sqrt_d = math.sqrt(d)
    root1 = (-b - sqrt_d) / (2 * a)
    root2 = (-b + sqrt_d) / (2 * a)
    return (float(root1), float(root2))

def new_quadratic_function(a, b, c):
    """Create and return a new, anonymous function (for example
    using a lambda expression) that takes one argument x and 
    returns the value of ax^2 + bx + c."""
    return lambda x: a * x ** 2 + b * x + c

def perfect_shuffle(even_list):
    """Assume even_list is a list of an even number of elements.
    Return a new list that is the perfect-shuffle of the input.
    Perfect shuffle means splitting a list into two halves and then interleaving
    them. For example, the perfect shuffle of [0, 1, 2, 3, 4, 5, 6, 7] is
    [0, 4, 1, 5, 2, 6, 3, 7]."""
    n = len(even_list)
    half = n // 2
    first_half = even_list[:half]
    second_half = even_list[half:]
    result = []
    for a, b in zip(first_half, second_half):
        result.append(a)
        result.append(b)
    return result

def list_of_5_times_elts_plus_1(input_list):
    """Assume a list of numbers is input. Using a list comprehension,
    return a new list in which each input element has been multiplied
    by 5 and had 1 added to it."""
    return [5 * x + 1 for x in input_list]

def double_vowels(text):
    """Return a new version of text, with all the vowels doubled.
    For example:  "The *BIG BAD* wolf!" => "Theee "BIIG BAAD* woolf!".
    For this exercise assume the vowels are
    the characters A,E,I,O, and U (and a,e,i,o, and u).
    Maintain the case of the characters."""
    vowels = "AEIOUaeiou"
    new_text = ""
    for char in text:
        if char in vowels:
            new_text += char * 2
        else:
            new_text += char
    return new_text

def count_words(text):
    """Return a dictionary having the words in the text as keys,
    and the numbers of occurrences of the words as values.
    Assume a word is a substring of letters and digits and the characters
    '-', '+', *', '/', '@', '#', '%', and "'" separated by whitespace,
    newlines, and/or punctuation (characters like . , ; ! ? & ( ) [ ] { } | : ).
    Convert all the letters to lower-case before the counting."""

    pattern = r"[A-Za-z0-9\-\+\*\/@#%']+"
    text_lower = text.lower()
    words = re.findall(pattern, text_lower)
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts


class TTT_State:
    
    def __init__(self):
        '''Create an instance. This happens to represent the initial state
        for Tic-Tac-Toe.'''
        self.board = [[" ", " ", " "],
                      [" ", " ", " "],
                      [" ", " ", " "]]
        self.whose_move = 'X'

    def __str__(self):
        '''Return a string representation of the
        state that show the Tic-Tac-Toe board as a 2-D ASCII display.
        Style it simply, as you wish.'''
        rows = []
        for row in self.board:
            rows.append(" | ".join(row))
        return "\n---------\n".join(rows)

    def __deepcopy__(self):
        '''Return a new instance with the same board arrangement 
        and player to move. 
        (Sublists must be copies, not copies of references.)'''
        new_state = TTT_State()
        new_state.board = [row[:] for row in self.board]
        new_state.whose_move = self.whose_move
        return new_state

    def __eq__(self, other):
        '''Return True iff two states are equal.'''
        if not isinstance(other, TTT_State):
            return False
        return self.board == other.board and self.whose_move == other.whose_move

class TTT_Operator:
    '''An instance of this class will represent an
    operator that can make a move by who (either 'X' or 'O'),
    to the given row and column. '''
    
    def __init__(self, who, row, col):
        self.who = who
        self.row = row
        self.col = col
    
    def is_applicable(self, state):
        '''Return True iff it would be legal to apply
        this operator to the given state.'''
        if not isinstance(state, TTT_State):
            return False
        if state.whose_move != self.who:
            return False
        if not (0 <= self.row < 3 and 0 <= self.col < 3):
            return False
        if state.board[self.row][self.col] != " ":
            return False
        return True

    def apply(self, state):
        '''Return a new state object that represents the
        result of applying this operator to the given state.'''
        new_state = TTT_State()
        new_state.board = [row[:] for row in state.board]
        new_state.whose_move = state.whose_move
        new_state.board[self.row][self.col] = self.who
        new_state.whose_move = 'O' if self.who == 'X' else 'X'
        return new_state
