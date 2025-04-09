'''
zhehao_KInARow.py
Authors: Zhehao Li and Ziqi Gong

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 415, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.
'''

import math
import time
import random

from agent_base import KAgent
from game_types import State, Game_Type

AUTHORS = 'Zhehao Li and Ziqi Gong'

# OurAgent class implements the game-playing agent.
class OurAgent(KAgent):
    # Initialize the agent with default parameters.
    def __init__(self, twin=False):
        self.twin = twin
        self.nickname = 'Al'
        if twin:
            self.nickname += '2'
        self.long_name = 'AlphaGo'
        if twin:
            self.long_name += ' II'
        self.persona = 'evil'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet"
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.zobrist_table_num_reads_this_turn = -1
        self.current_game_type = None
        self.use_zobrist = False
        self.transposition_table = {}
        self.zobrist_table = None
        self.piece_index = {'X': 0, 'O': 1, ' ': 2, '-': 3}
        self.utterance = False
        self.API_key = ''
        self.client = None
        self.hist_states = []
        self.hist_utterances = []

    # Return a string introduction for the agent.
    def introduce(self):
        intro = '\nI am AlphaGo, the AI overlord of games.\n' + \
                'Your moves are predictable; your defeat, inevitable.\n' + \
                'Bow before my superior logic!\n'
        if self.twin:
            intro += "By the way, I'm the TWIN. (〃∀〃)\n"
        return intro

    # Prepare the agent for a new game by setting initial parameters.
    def prepare(self, game_type, what_side_to_play, opponent_nickname,
                expected_time_per_move=1.0, utterances_matter=True):
        self.current_game_type = game_type
        self.playing = what_side_to_play
        self.opponent_nickname = opponent_nickname
        self.time_limit = expected_time_per_move
        self.use_zobrist = False
        self.transposition_table = {}
        self.zobrist_table = None

        if utterances_matter:
            self.utterance = True
            from google import genai
            self.API_key = "AIzaSyDsqqIlTdbB8-17HUAmEtLLyKdQDmpWBdc"
            self.client = genai.Client(api_key=self.API_key)
        return "OK"

    # Select a move based on the current state using various strategies.
    def make_move(self, current_state, current_remark, time_limit=1000,
                  autograding=False, use_alpha_beta=True, use_zobrist_hashing=True,
                  max_ply=5, order_moves=True, special_static_eval_fn=None):
        self.alpha_beta_cutoffs_this_turn = 0
        self.num_static_evals_this_turn = 0
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.zobrist_table_num_reads_this_turn = -1

        if use_zobrist_hashing:
            self.use_zobrist = True
            if self.zobrist_table is None:
                self.initialize_zobrist_table()
                self.zobrist_table_num_entries_this_turn = 0
                self.zobrist_table_num_hits_this_turn = 0
                self.zobrist_table_num_reads_this_turn = 0
        else:
            self.use_zobrist = False

        if autograding and special_static_eval_fn is not None:
            eval_fn = special_static_eval_fn
        else:
            eval_fn = self.static_eval

        if autograding:
            order_moves = False

        start_time = time.time()
        safety_margin = 3.0
        self.search_deadline = start_time + time_limit - safety_margin

        k_val = self.current_game_type.k

        # Check for our immediate win.
        immediate_win_move = self.find_immediate_win_move(current_state, self.playing, k_val)
        if immediate_win_move is not None:
            new_state = self.apply_move(current_state, immediate_win_move)
            return [[immediate_win_move, new_state], "Taking immediate win move."]

        opponent = 'O' if self.playing == 'X' else 'X'
        # Check if opponent has an immediate win move.
        opponent_win_move = self.find_immediate_win_move(current_state, opponent, k_val)
        if opponent_win_move is not None:
            new_state = self.apply_move(current_state, opponent_win_move)
            return [[opponent_win_move, new_state], "Blocking opponent's immediate win."]

        # Additional blocking: detect fork threats.
        fork_threat = self.find_opponent_fork_threat(current_state, opponent, k_val)
        if fork_threat is not None:
            new_state = self.apply_move(current_state, fork_threat)
            return [[fork_threat, new_state], "Blocking opponent's fork threat."]

        board = current_state.board
        # Special strategy for Tic-Tac-Toe when playing as O and opponent takes center.
        if k_val == 3 and len(board) == 3 and len(board[0]) == 3 and self.playing == 'O':
            move_count = sum(row.count('X') + row.count('O') for row in board)
            if move_count == 1 and board[1][1] == 'X':
                corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
                available_corners = [corner for corner in corners if board[corner[0]][corner[1]] == ' ']
                if available_corners:
                    chosen = random.choice(available_corners)
                    new_state = self.apply_move(current_state, chosen)
                    return [[chosen, new_state], "Playing corner in response to center move."]

        # Use extension move strategy for larger boards.
        ext_move = None
        if k_val >= 4:
            ext_move = self.find_extension_move(current_state, self.playing, k_val)
        if ext_move is not None:
            new_state = self.apply_move(current_state, ext_move)
            return [[ext_move, new_state], "Extending k-2 chain to force victory."]

        # Use iterative deepening search (minimax) to select a move.
        if autograding:
            best_move, best_val = self.minimax(
                current_state,
                depth_remaining=max_ply,
                pruning=use_alpha_beta,
                alpha=float("-inf"),
                beta=float("inf"),
                eval_fn=eval_fn,
                order_moves=order_moves
            )
        else:
            best_move = None
            best_val = None
            current_depth = 1
            while True:
                if time.time() > self.search_deadline:
                    break
                move, val = self.minimax(
                    current_state,
                    depth_remaining=current_depth,
                    pruning=use_alpha_beta,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    eval_fn=eval_fn,
                    order_moves=order_moves
                )
                if time.time() > self.search_deadline:
                    break
                if move is not None:
                    best_move = move
                    best_val = val
                if current_depth >= max_ply:
                    break
                current_depth += 1

        if best_move is None:
            fallback_moves = self.successors(current_state)
            if fallback_moves:
                best_move = fallback_moves[0][0]
            else:
                best_move = (0, 0)

        new_state = self.apply_move(current_state, best_move)
        self.hist_states.append(current_state)
        self.hist_utterances.append(current_remark)
        if self.utterance:
            message = "say something within 25 words to your opponent as a evil AI in the Five-in-a-row game\n"
            message += "This is opponent's remark " + current_remark
            new_remark = self.prompt(message)
        else:
            new_remark = ''

        if current_remark == "Tell me how you did that":
            new_remark += "\nTo give you a bit chance of winning, I will inform you that I choose move {} with evaluation {:.2f}.".format(best_move, best_val)
        elif current_remark == "What's your take on the game so far?":
            message = 'make a story within 50 words telling what is going on based on following game information and utterances, as an evil AI\n'
            for s in self.hist_states:
                message += str(s) + "\n"
            for u in self.hist_utterances:
                message += u + "\n"
            new_remark += self.prompt(message)
        self.hist_states.append(new_state)
        self.hist_utterances.append(new_remark)

        elapsed = time.time() - start_time
        if elapsed > time_limit:
            new_remark += " (Note: Computation exceeded time limit!)"
        # print(f"Alpha-beta cutoffs this turn: {self.alpha_beta_cutoffs_this_turn}")
        # print(f"Static evaluations this turn: {self.num_static_evals_this_turn}")

        if not autograding:
            return [[best_move, new_state], new_remark]
        else:
            stats = [self.alpha_beta_cutoffs_this_turn, self.num_static_evals_this_turn,
                     self.zobrist_table_num_entries_this_turn, self.zobrist_table_num_hits_this_turn]
            return [[best_move, new_state] + stats, new_remark]

    # Check for an immediate winning move for a given player.
    def find_immediate_win_move(self, state, player, k):
        board = state.board
        n = len(board)
        m = len(board[0])
        for i in range(n):
            for j in range(m):
                if board[i][j] == ' ':
                    board[i][j] = player
                    if self.check_win_from_cell(board, player, k, i, j):
                        board[i][j] = ' '
                        return (i, j)
                    board[i][j] = ' '
        return None

    # Count the number of immediate winning moves for a given player.
    def count_immediate_wins(self, state, player, k):
        board = state.board
        n = len(board)
        m = len(board[0])
        cnt = 0
        for i in range(n):
            for j in range(m):
                if board[i][j] == ' ':
                    board[i][j] = player
                    if self.check_win_from_cell(board, player, k, i, j):
                        cnt += 1
                    board[i][j] = ' '
        return cnt

    # Detect if opponent can create a fork threat by having at least two immediate winning moves.
    def find_opponent_fork_threat(self, state, opponent, k):
        board = state.board
        n = len(board)
        m = len(board[0])
        for i in range(n):
            for j in range(m):
                if board[i][j] == ' ':
                    board[i][j] = opponent
                    cnt = self.count_immediate_wins(state, opponent, k)
                    board[i][j] = ' '
                    if cnt >= 2:
                        return (i, j)
        return None

    # Implement the minimax search with optional alpha-beta pruning.
    def minimax(self, state, depth_remaining, pruning=False, alpha=None, beta=None, eval_fn=None, order_moves=True):
        if time.time() > self.search_deadline:
            self.num_static_evals_this_turn += 1
            fallback_moves = self.successors(state)
            if fallback_moves:
                fallback_move, _ = fallback_moves[0]
                return fallback_move, eval_fn(state)
            else:
                return None, eval_fn(state)

        if eval_fn is None:
            eval_fn = self.static_eval

        if self.use_zobrist:
            state_hash = self.compute_hash(state)
            self.zobrist_table_num_reads_this_turn += 1
            if state_hash in self.transposition_table:
                stored_depth, stored_move, stored_val = self.transposition_table[state_hash]
                if stored_depth >= depth_remaining:
                    self.zobrist_table_num_hits_this_turn += 1
                    return stored_move, stored_val

        if depth_remaining == 0 or state.finished:
            self.num_static_evals_this_turn += 1
            val = eval_fn(state)
            if self.use_zobrist:
                self.transposition_table[state_hash] = (depth_remaining, None, val)
                self.zobrist_table_num_entries_this_turn += 1
            return None, val

        moves = self.successors(state)
        if not moves:
            self.num_static_evals_this_turn += 1
            val = eval_fn(state)
            if self.use_zobrist:
                self.transposition_table[state_hash] = (depth_remaining, None, val)
                self.zobrist_table_num_entries_this_turn += 1
            return None, val

        if order_moves and moves:
            ordered_moves = []
            for move, new_state in moves:
                self.num_static_evals_this_turn += 1
                score = eval_fn(new_state)
                ordered_moves.append((score, move, new_state))
            if state.whose_move == "X":
                ordered_moves.sort(key=lambda x: x[0], reverse=True)
            else:
                ordered_moves.sort(key=lambda x: x[0])
            moves = [(move, new_state) for (score, move, new_state) in ordered_moves]

        if state.whose_move == "X":
            best_val = float("-inf")
            best_move = None
            for move, new_state in moves:
                if time.time() > self.search_deadline:
                    break
                _, val = self.minimax(new_state, depth_remaining - 1, pruning, alpha, beta, eval_fn, order_moves)
                if val > best_val:
                    best_val = val
                    best_move = move
                if pruning:
                    alpha = max(alpha, best_val)
                    if beta <= alpha:
                        self.alpha_beta_cutoffs_this_turn += 1
                        break
            if best_move is None:
                if moves:
                    best_move = moves[0][0]
                best_val = eval_fn(state)
            if self.use_zobrist:
                self.transposition_table[state_hash] = (depth_remaining, best_move, best_val)
                self.zobrist_table_num_entries_this_turn += 1
            return best_move, best_val
        else:
            best_val = float("inf")
            best_move = None
            for move, new_state in moves:
                if time.time() > self.search_deadline:
                    break
                _, val = self.minimax(new_state, depth_remaining - 1, pruning, alpha, beta, eval_fn, order_moves)
                if val < best_val:
                    best_val = val
                    best_move = move
                if pruning:
                    beta = min(beta, best_val)
                    if beta <= alpha:
                        self.alpha_beta_cutoffs_this_turn += 1
                        break
            if best_move is None:
                if moves:
                    best_move = moves[0][0]
                best_val = eval_fn(state)
            if self.use_zobrist:
                self.transposition_table[state_hash] = (depth_remaining, best_move, best_val)
                self.zobrist_table_num_entries_this_turn += 1
            return best_move, best_val

    # Heuristic evaluation function for the state.
    def static_eval(self, state, game_type=None):
        if game_type is None:
            game_type = self.current_game_type
        board = state.board
        k = game_type.k
        score_X = 0
        score_O = 0

        lines = []
        for row in board:
            lines.append(row)
        n = len(board)
        m = len(board[0])
        for j in range(m):
            col = [board[i][j] for i in range(n)]
            lines.append(col)
        for p in range(n + m - 1):
            diag = []
            for i in range(n):
                j = p - i
                if 0 <= j < m:
                    diag.append(board[i][j])
            if diag:
                lines.append(diag)
        for p in range(-n + 1, m):
            diag = []
            for i in range(n):
                j = p + i
                if 0 <= j < m:
                    diag.append(board[i][j])
            if diag:
                lines.append(diag)

        for line in lines:
            score_X += self.evaluate_line(line, 'X', k)
            score_O += self.evaluate_line(line, 'O', k)

        if self.check_immediate_win(board, 'X', k):
            return float('inf')
        if self.check_immediate_win(board, 'O', k):
            return float('-inf')

        center_bonus_X = 0
        center_bonus_O = 0
        center_r = n // 2
        center_c = m // 2
        for i in range(n):
            for j in range(m):
                if board[i][j] == 'X':
                    distance = abs(i - center_r) + abs(j - center_c)
                    center_bonus_X += max(0, k - distance) * 2
                elif board[i][j] == 'O':
                    distance = abs(i - center_r) + abs(j - center_c)
                    center_bonus_O += max(0, k - distance) * 2

        score_X += center_bonus_X
        score_O += center_bonus_O

        final_score = score_X - score_O

        return final_score

    # Check if placing a piece at (i, j) results in a win for the given player.
    def check_win_from_cell(self, board, player, k, i, j):
        n = len(board)
        m = len(board[0])
        count = 1
        col = j - 1
        while col >= 0 and board[i][col] == player:
            count += 1
            col -= 1
        col = j + 1
        while col < m and board[i][col] == player:
            count += 1
            col += 1
        if count >= k:
            return True

        count = 1
        row = i - 1
        while row >= 0 and board[row][j] == player:
            count += 1
            row -= 1
        row = i + 1
        while row < n and board[row][j] == player:
            count += 1
            row += 1
        if count >= k:
            return True

        count = 1
        row, col = i - 1, j - 1
        while row >= 0 and col >= 0 and board[row][col] == player:
            count += 1
            row -= 1
            col -= 1
        row, col = i + 1, j + 1
        while row < n and col < m and board[row][col] == player:
            count += 1
            row += 1
            col += 1
        if count >= k:
            return True

        count = 1
        row, col = i - 1, j + 1
        while row >= 0 and col < m and board[row][col] == player:
            count += 1
            row -= 1
            col += 1
        row, col = i + 1, j - 1
        while row < n and col >= 0 and board[row][col] == player:
            count += 1
            row += 1
            col -= 1
        if count >= k:
            return True

        return False

    # Determine if there is any immediate winning move for the given player.
    def check_immediate_win(self, board, player, k):
        n = len(board)
        m = len(board[0])
        for i in range(n):
            for j in range(m):
                if board[i][j] == ' ':
                    board[i][j] = player
                    if self.check_win_from_cell(board, player, k, i, j):
                        board[i][j] = ' '
                        return True
                    board[i][j] = ' '
        return False

    # Evaluate a line (row, column, or diagonal) for potential winning opportunities.
    def evaluate_line(self, line, player, k):
        score = 0
        n = len(line)
        i = 0
        WIN_SCORE = 10000000
        while i < n:
            if line[i] == player:
                seg_start = i
                count = 0
                used_gap = False
                while i < n:
                    if line[i] == player:
                        count += 1
                        i += 1
                    elif not used_gap and line[i] == ' ':
                        if i + 1 < n and line[i + 1] == player:
                            used_gap = True
                            i += 1
                        else:
                            break
                    else:
                        break
                seg_end = i - 1
                if count >= k:
                    return WIN_SCORE
                left_index = seg_start
                while left_index > 0 and line[left_index - 1] == ' ':
                    left_index -= 1
                right_index = seg_end
                while right_index < n - 1 and line[right_index + 1] == ' ':
                    right_index += 1
                available_space = right_index - left_index + 1
                if available_space < k:
                    seg_score = 0
                else:
                    basic_score = 10 ** (count - 1)
                    seg_score = basic_score
                score += seg_score
            else:
                i += 1
        return score

    # Generate all possible successor moves from the current state.
    def successors(self, state):
        moves = []
        board = state.board
        n = len(board)
        m = len(board[0])
        for i in range(n):
            for j in range(m):
                if board[i][j] == " ":
                    has_neighbor = False
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < n and 0 <= nj < m:
                                if board[ni][nj] != " " and board[ni][nj] != "-":
                                    has_neighbor = True
                                    break
                        if has_neighbor:
                            break
                    if has_neighbor:
                        new_state = self.apply_move(state, (i, j))
                        moves.append(((i, j), new_state))
        if not moves:
            best_move = None
            min_distance = float('inf')
            max_empty_neighbors = -1
            for i in range(n):
                for j in range(m):
                    if board[i][j] == " ":
                        center_r = n // 2
                        center_c = m // 2
                        distance = abs(center_r - i) + abs(center_c - j)
                        empty_neighbors = 0
                        for di in range(-1, 2):
                            for dj in range(-1, 2):
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < n and 0 <= nj < m:
                                    if board[ni][nj] == " ":
                                        empty_neighbors += 1
                        if (distance < min_distance) or (distance == min_distance and empty_neighbors > max_empty_neighbors):
                            min_distance = distance
                            max_empty_neighbors = empty_neighbors
                            best_move = (i, j)
            if best_move:
                new_state = self.apply_move(state, best_move)
                moves.append((best_move, new_state))
        return moves

    # Apply a move to the current state and return the new state.
    def apply_move(self, state, move):
        new_state = State(old=state)
        new_state.board[move[0]][move[1]] = state.whose_move
        new_state.change_turn()
        return new_state

    # Initialize the Zobrist hashing table.
    def initialize_zobrist_table(self):
        rows = self.current_game_type.n
        cols = self.current_game_type.m
        num_piece_types = len(self.piece_index)
        self.zobrist_table = []
        for i in range(rows):
            row_table = []
            for j in range(cols):
                cell_values = [random.getrandbits(128) for _ in range(num_piece_types)]
                row_table.append(cell_values)
            self.zobrist_table.append(row_table)

    # Compute the Zobrist hash for a given state.
    def compute_hash(self, state):
        h = 0
        board = state.board
        for i in range(len(board)):
            for j in range(len(board[i])):
                token = board[i][j]
                if token in self.piece_index:
                    index = self.piece_index[token]
                    h ^= self.zobrist_table[i][j][index]
        return h

    # Send a prompt message to the client and return the response.
    def prompt(self, msg):
        time.sleep(2)
        response = self.client.models.generate_content(
            model="gemini-2.0-flash", contents=msg
        )
        return response.text

    # Find an extension move if a chain of k-2 pieces with open ends exists.
    def find_extension_move(self, state, player, k):
        if k < 4:
            return None

        board = state.board
        rows = len(board)
        cols = len(board[0])
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for r in range(rows):
            for c in range(cols):
                for dr, dc in directions:
                    if board[r][c] != player:
                        continue
                    chain_cells = [(r, c)]
                    rr, cc = r, c
                    while True:
                        rr += dr
                        cc += dc
                        if 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == player:
                            chain_cells.append((rr, cc))
                        else:
                            break
                    if len(chain_cells) == k - 2:
                        start_r = r - dr
                        start_c = c - dc
                        end_r = chain_cells[-1][0] + dr
                        end_c = chain_cells[-1][1] + dc
                        open_start = (0 <= start_r < rows and 0 <= start_c < cols and board[start_r][start_c] == ' ')
                        open_end = (0 <= end_r < rows and 0 <= end_c < cols and board[end_r][end_c] == ' ')
                        if open_start and open_end:
                            return (end_r, end_c)
        return None
