import ast
import inspect
import types
import unittest


import a1_exercises as a1

# Helper function for 2 tests.
def eq_test(s1, s2):
    for rn in range(3):
         for cn in range(3):
             if s1.board[rn][cn] != s2.board[rn][cn]: return False
    if s1.whose_move != s2.whose_move: return False
    return True
        

class TestA1Functions(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    def _test_function_on_cases(self, func, test_cases, test_type=False,
                                expect_type=None, assert_fn=None):
        if assert_fn is None:
            assert_fn = self.assertEqual

        for test_input, expected_output in test_cases:
            actual_output = func(test_input)
            assert_fn(
                expected_output, actual_output,
                f"{func.__name__}({test_input!r}) \n"
                f"returned: {actual_output!r}\n"
                f"expected: {expected_output!r}")
            if test_type:
                self.assertIs(
                    type(actual_output), expect_type,
                    f"Return value of {func.__name__}({test_input!r}) is "
                    f"not of type {expect_type.__name__}!")

    def test_is_quintuple(self):
        """Provided tests for is_triple in starter code."""
        test_cases = [
            (0, True), (145, True), (1, False), (22, False), (-1, False),
            (-15, True),
        ]
        self._test_function_on_cases(a1.is_a_quintuple, test_cases)

    def test_last_prime(self):
        """Provided tests for last_prime in starter code."""
        test_cases = [
            (2, 2), (3, 3), (4, 3), (10, 7), (12, 11), (100, 97),
            (200, 199),
        ]
        self._test_function_on_cases(a1.last_prime, test_cases, test_type=True,
                                     expect_type=int)

    def test_quadratic_roots(self):
        """Provided tests for quadratic_roots in starter code."""
        roots = a1.quadratic_roots(1, 4, -21)
        assert type(roots) is tuple, (
            "Value returned from quadratic_roots(1, 4, -21) is not a tuple!")
        x1, x2 = sorted(roots)
        assert type(x1) is float and type(x2) is float, (
            "Roots returned from quadratic_roots(1, 4, -21) are not floats!")
        self.assertAlmostEqual(x1, -7.0)
        self.assertAlmostEqual(x2, 3.0)

        self.assertEqual(a1.quadratic_roots(1, 1, 1), "complex")

    def test_new_quadratic_function(self):
        """Provided tests for new_quadratic_function in starter code."""
        qf1 = a1.new_quadratic_function(3, 2, 1)
        qf2 = a1.new_quadratic_function(6, 5, 4)
        result = qf1(10)
        assert(result == 321)
        result = qf2(10)
        assert(result == 654)

    def test_perfect_shuffle(self):
        """Provided tests for perfect_shuffle in starter code."""
        test_cases = [
            ([], []), ([1, 2, 3, 4], [1, 3, 2, 4]),
            ([0, 1, 2, 3, 4, 5, 6, 7], [0, 4, 1, 5, 2, 6, 3, 7]),
        ]
        self._test_function_on_cases(a1.perfect_shuffle, test_cases)

    def test_list_of_5_times_elts_plus_1(self):
        """Provided tests for list_of_3_times_elts_plus_1 in starter code."""
        test_cases = [([], []), ([1], [6]), ([1, 2, 3], [6, 11, 16])]
        self._test_function_on_cases(a1.list_of_5_times_elts_plus_1, test_cases)

    def test_list_of_5_times_elts_plus_1_use_list_comp(self):
        """Provided test for list_of_5_times_elts_plus_1 on list comprehension."""
        the_ast = ast.parse(inspect.getsource(a1.list_of_5_times_elts_plus_1))
        used_list_comp = any(
            type(node) is ast.ListComp for node in
            ast.walk(the_ast))
        self.assertTrue(
            used_list_comp, "Your did not use list comprehension when "
                            "implementing list_of_5_times_elts_plus_1!")

    def test_double_vowels(self):
        """Provided tests for double_vowels in starter code."""
        test_cases = [
            ("The big bad WOLF", "Thee biig baad WOOLF"),
            ("The *BIG BAD* wolf!", "Thee *BIIG BAAD* woolf!"),
        ]
        self._test_function_on_cases(a1.double_vowels, test_cases)

    def test_count_words(self):
        """Provided test for count_words in starter code."""
        test_cases = [
            # Small test cases.
            (" A a a b b ", {"a": 3, "b": 2}),
            ("#screen-size: 1920*1080, 2560*1440, 1920*1080",
             {'#screen-size': 1, '1920*1080': 2, '2560*1440': 1}),
            # Natural text test cases.
            ("""Don't lie
                I want him to know
                Gods' loves die young
                Is he ready to go?
                It's the last time running through snow
                Where the vaults are full and the fire's bold
                I want to know - does it bother you?
                The low click of a ticking clock
                There's a lifetime right in front of you
                And everyone I know
                Young turks
                Young saturday night
                Young hips shouldn't break on this ice
                Old flames - they can't warm you tonight""",
             {"don't": 1, 'lie': 1, 'i': 3, 'want': 2, 'him': 1, 'to': 3,
              'know': 3, "gods'": 1, 'loves': 1, 'die': 1, 'young': 4, 'is': 1,
              'he': 1, 'ready': 1, 'go': 1, "it's": 1, 'the': 4, 'last': 1,
              'time': 1, 'running': 1, 'through': 1, 'snow': 1, 'where': 1,
              'vaults': 1, 'are': 1, 'full': 1, 'and': 2, "fire's": 1,
              'bold': 1, '-': 2, 'does': 1, 'it': 1, 'bother': 1, 'you': 3,
              'low': 1, 'click': 1, 'of': 2, 'a': 2, 'ticking': 1, 'clock': 1,
              "there's": 1, 'lifetime': 1, 'right': 1, 'in': 1, 'front': 1,
              'everyone': 1, 'turks': 1, 'saturday': 1, 'night': 1, 'hips': 1,
              "shouldn't": 1, 'break': 1, 'on': 1, 'this': 1, 'ice': 1,
              'old': 1, 'flames': 1, 'they': 1, "can't": 1, 'warm': 1,
              'tonight': 1}),
        ]
        self._test_function_on_cases(
            a1.count_words, test_cases, test_type=True, expect_type=dict,
            assert_fn=self.assertDictEqual)

    def test_TTT_State_eq(self):
        """Provided tests for the equality method in TTT_State in starter code."""
        s1 = a1.TTT_State()
        s2 = a1.TTT_State()
        s3 = a1.TTT_State()
        s3.board[0][0] = 'X'
        test_cases = [
            (s2, True),
            (s3, False),
        ]
        self._test_function_on_cases(s1.__eq__, test_cases)

    def test_TTT_State_deepcopy(self):
        """Provided tests for the deepcopy method in TTT_State in starter code.
        To pass the test, 3 conditions must be satisfied:
        a. The strings associated with the 2 states must be equal.
        b. The object references must be different (the id's).
        c. Changing the deep copy must make the orig and copy no longer equal.

        Rather than directly test __deepcopy__ we will pass a special test
          function to the standard tester."""

        s1 = a1.TTT_State()
        s1.board[0][0] = 'X'
        s2 = s1.__deepcopy__()
        def dc_test(num):
            if num==1:
                return s1 is s2 # Should be False
            if num==2:
                return eq_test(s1, s2)
            if num==3:
                s2.board[0][1]='O'
                return '0'==s1.board[0][1] # Should be False
        
        test_cases = [
            (1, False), (2, True), (3, False),
        ]
        self._test_function_on_cases(dc_test, test_cases)

    def test_TTT_Operator_is_applicable(self):
        """Provided tests for the is_applicable method in TTT_Operator in starter code."""
        s1 = a1.TTT_State()
        s2 = a1.TTT_State()
        s2.board[0][0] = 'X'
        x_in_00 = a1.TTT_Operator('X', 0, 0)
        test_cases = [
            (s1, True),
            (s2, False),
        ]
        self._test_function_on_cases(x_in_00.is_applicable, test_cases)

    def test_TTT_Operator_apply(self):
        """Provided tests for the apply method in TTT_Operator in starter code."""
        s1 = a1.TTT_State()  # Initial state
        s2 = a1.TTT_State()  # another instance of it
        s2.board[0][0] = 'X' # Making the move directly here.
        s2.whose_move = 'O'  #   "
        x_in_00 = a1.TTT_Operator('X', 0, 0)
        def apply_test(num):
            if num==1:
                s3 = x_in_00.apply(s1)
                return eq_test(s2, s3)
            
        test_cases = [
            (1, True),
        ]
        self._test_function_on_cases(apply_test, test_cases)


if __name__ == '__main__':
    unittest.main()
