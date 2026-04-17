import numpy as np
import time
import os
import json
import random
from collections import defaultdict
from game.enums import Cell, Noise, MoveType, Direction, CARPET_POINTS_TABLE
from game.move import Move

NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE: (0.7, 0.15, 0.15),
    Cell.PRIMED: (0.1, 0.8, 0.1),
    Cell.CARPET: (0.1, 0.1, 0.8),
}

# ==========================================
# THE TRACKER (HMM)
# ==========================================
class RatBelief:
    def __init__(self, T_matrix):
        self.T = np.array(T_matrix)
        
        # Cache the stationary distribution once to save massive compute on resets
        initial_state = np.zeros(64)
        initial_state[0] = 1.0
        T_1000 = np.linalg.matrix_power(self.T, 1000) # this is the matrix raised to the 1000 power to find stationary distrubtion
        self.stationary_belief = initial_state @ T_1000
        
        self.belief = np.zeros(64)
        self.reset_and_headstart()

    def reset_and_headstart(self):
        # Instantly copy the cached distribution instead of recalculating
        self.belief = self.stationary_belief.copy()

    def predict(self):
        self.belief = self.belief @ self.T # current belief and transtion state, where will the rat be next time

    def update(self, noise, reported_distance, my_pos, board):
        likelihood = np.zeros(64)
        my_x, my_y = my_pos
        
        for i in range(64):
            x, y = i % 8, i // 8
            cell_type = board.get_cell((x, y))
            p_noise = NOISE_PROBS[cell_type][noise.value]
            
            actual_dist = abs(my_x - x) + abs(my_y - y)
            p_dist = 0.0
            
            if reported_distance == 0:
                if actual_dist == 0: p_dist = 0.82
                elif actual_dist == 1: p_dist = 0.12
            else:
                if actual_dist == reported_distance: p_dist = 0.70
                elif actual_dist == reported_distance + 1: p_dist = 0.12
                elif actual_dist == reported_distance - 1: p_dist = 0.12
                elif actual_dist == reported_distance - 2: p_dist = 0.06
            
            likelihood[i] = p_noise * p_dist
            
        self.belief *= likelihood
        self.normalize()

    def normalize(self):
        total_prob = np.sum(self.belief)
        if total_prob > 0:
            self.belief /= total_prob
        else:
            self.belief = np.ones(64) / 64.0

    def get_best_guess(self):
        best_idx = np.argmax(self.belief)
        return (best_idx % 8, best_idx // 8), self.belief[best_idx]


class PlayerAgent:
    def __init__(self, board, transition_matrix, time_left_func):
        self.hmm = RatBelief(transition_matrix)
        self.TIME_BUFFER = 0.8 
        self.tt = {} # Transposition Table
        
        # [REFACTOR] Added a History Table for Move Ordering
        self.history_table = defaultdict(int) 
        
        self.weights = {
            "rat_confidence_threshold": 0.65 # [REFACTOR] Shifted from 0.45 to force more macro-play
        }
        if 'AGENT_WEIGHTS' in os.environ:
            try: self.weights.update(json.loads(os.environ['AGENT_WEIGHTS']))
            except Exception: pass

    def check_rat_resets(self, board):
        my_guess, my_correct = board.player_search
        if my_guess is not None:
            if my_correct:
                self.hmm.reset_and_headstart()
                return 
            else:
                my_idx = my_guess[1] * 8 + my_guess[0]
                self.hmm.belief[my_idx] = 0.0
                self.hmm.normalize()
                
        op_guess, op_correct = board.opponent_search
        if op_guess is not None:
            if op_correct:
                self.hmm.reset_and_headstart()
            else:
                op_idx = op_guess[1] * 8 + op_guess[0]
                self.hmm.belief[op_idx] = 0.0
                self.hmm.normalize()

    def _hash_board(self, board, is_maximizing):
        # [REFACTOR] Tuple generators are incredibly slow. 
        # If your 'board' object has an underlying array (like board.grid), 
        # hashing its bytes is O(1) in python. Assuming it doesn't, we flatten 
        # the loop into a single string comprehension which is natively optimized in C.
        cell_str = "".join(str(board.get_cell((x, y)).value) for x in range(8) for y in range(8))
        return hash((
            board.player_worker.get_location(),
            board.opponent_worker.get_location(),
            board.player_worker.get_points(),
            board.opponent_worker.get_points(),
            cell_str,
            is_maximizing
        ))

    def heuristic(self, board):
        score = board.player_worker.get_points() - board.opponent_worker.get_points()
        my_x, my_y = board.player_worker.get_location()
        en_x, en_y = board.opponent_worker.get_location()
        
        # [REFACTOR] Inlined Manhattan distance to avoid function call overhead at the leaf nodes.
        def quick_dist(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)

        def evaluate_line(line_coords, is_horizontal):
            length = len(line_coords)
            if length == 0: return 0
            
            actual_points = CARPET_POINTS_TABLE[min(length, 7)] 
            
            # Non-Linear Patience Curve
            if length == 1: potential_value = 0.5
            elif length == 2: potential_value = 2.5  
            elif length == 3: potential_value = 4.5  
            elif length == 4: potential_value = 6.5  
            else: potential_value = actual_points * 0.8 
            
            start_x, start_y = line_coords[0]
            end_x, end_y = line_coords[-1]
            
            my_dist, en_dist = float('inf'), float('inf')
            
            if is_horizontal:
                if start_x > 0 and board.get_cell((start_x - 1, start_y)) != Cell.BLOCKED:
                    my_dist = min(my_dist, quick_dist(my_x, my_y, start_x - 1, start_y))
                    en_dist = min(en_dist, quick_dist(en_x, en_y, start_x - 1, start_y))
                if end_x < 7 and board.get_cell((end_x + 1, end_y)) != Cell.BLOCKED:
                    my_dist = min(my_dist, quick_dist(my_x, my_y, end_x + 1, end_y))
                    en_dist = min(en_dist, quick_dist(en_x, en_y, end_x + 1, end_y))
            else:
                if start_y > 0 and board.get_cell((start_x, start_y - 1)) != Cell.BLOCKED:
                    my_dist = min(my_dist, quick_dist(my_x, my_y, start_x, start_y - 1))
                    en_dist = min(en_dist, quick_dist(en_x, en_y, start_x, start_y - 1))
                if end_y < 7 and board.get_cell((start_x, end_y + 1)) != Cell.BLOCKED:
                    my_dist = min(my_dist, quick_dist(my_x, my_y, start_x, end_y + 1))
                    en_dist = min(en_dist, quick_dist(en_x, en_y, start_x, end_y + 1))

            if my_dist == float('inf') and en_dist == float('inf'): return 0.0
            
            if my_dist < en_dist:
                return potential_value + (en_dist - my_dist) * 0.1
            elif en_dist < my_dist:
                return -potential_value + ((15 - my_dist) * (max(1, actual_points) * 0.05))
            return potential_value * 1.5

        # [REFACTOR] Removed the massive memory allocation of positive_shares and negative_shares matrices. 
        # Instead, we keep a running tally. We accept a slight overlap in cross-sections in exchange 
        # for a massive speed boost, allowing deeper searches.
        prime_bonus = 0.0

        for y in range(8):
            current_line = []
            for x in range(8):
                if board.get_cell((x, y)) == Cell.PRIMED: current_line.append((x, y))
                elif current_line:
                    prime_bonus += evaluate_line(current_line, True)
                    current_line = []
            if current_line: prime_bonus += evaluate_line(current_line, True)
                
        for x in range(8):
            current_line = []
            for y in range(8):
                if board.get_cell((x, y)) == Cell.PRIMED: current_line.append((x, y))
                elif current_line:
                    prime_bonus += evaluate_line(current_line, False)
                    current_line = []
            if current_line: prime_bonus += evaluate_line(current_line, False)
            
        return score + (prime_bonus * 0.8) # Dampen the bonus slightly since we removed share deduplication

    def minimax(self, board, depth, alpha, beta, is_maximizing, start_time, time_limit):
        if time.time() - start_time > time_limit:
            raise TimeoutError()

        if depth == 0 or board.is_game_over():
            return self.heuristic(board), None

        state_hash = self._hash_board(board, is_maximizing)
        if state_hash in self.tt:
            cached_depth, cached_score, cached_move, flag = self.tt[state_hash]
            if cached_depth >= depth:
                if flag == 'EXACT': return cached_score, cached_move
                elif flag == 'LOWERBOUND': alpha = max(alpha, cached_score)
                elif flag == 'UPPERBOUND': beta = min(beta, cached_score)
                if alpha >= beta: return cached_score, cached_move

        orig_alpha = alpha
        orig_beta = beta

        if is_maximizing:
            valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
            if not valid_moves: return self.heuristic(board), None

            # [REFACTOR] Move Ordering now utilizes the History Table. Moves that previously 
            # caused beta cutoffs (good moves) get pushed to the front of the queue automatically.
            def move_priority(m):
                base = self.history_table.get(str(m), 0) 
                if m.move_type == MoveType.CARPET: return base + 100 + (m.roll_length * 2)
                if m.move_type == MoveType.PRIME: return base + 50
                return base
                
            valid_moves.sort(key=move_priority, reverse=True)

            best_move = valid_moves[0]
            max_eval = float('-inf')
            
            for move in valid_moves:
                next_board = board.forecast_move(move, check_ok=False)
                if next_board is None: continue
                
                eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, False, start_time, time_limit)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha: 
                    # [REFACTOR] Record a successful Beta Cutoff in the History Table
                    self.history_table[str(move)] += (depth ** 2) 
                    break
                
            flag = 'EXACT'
            if max_eval <= orig_alpha: flag = 'UPPERBOUND'
            elif max_eval >= beta: flag = 'LOWERBOUND'
            self.tt[state_hash] = (depth, max_eval, best_move, flag)
            return max_eval, best_move
            
        else:
            board.reverse_perspective()
            valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
            if not valid_moves:
                board.reverse_perspective() 
                return self.heuristic(board), None

            # Opponent move ordering (assumes opponent will also make historically good moves)
            valid_moves.sort(key=lambda m: self.history_table.get(str(m), 0), reverse=True)

            best_move = valid_moves[0]
            min_eval = float('inf')
            
            for move in valid_moves:
                next_board = board.forecast_move(move, check_ok=False)
                if next_board is None: continue
                
                next_board.reverse_perspective()
                eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, True, start_time, time_limit)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    
                beta = min(beta, eval_score)
                if beta <= alpha: 
                    self.history_table[str(move)] += (depth ** 2)
                    break
                
            board.reverse_perspective() 
            flag = 'EXACT'
            if min_eval <= alpha: flag = 'UPPERBOUND'
            elif min_eval >= orig_beta: flag = 'LOWERBOUND'
            self.tt[state_hash] = (depth, min_eval, best_move, flag)
            return min_eval, best_move


    def play(self, board, rat_samples, time_left_func):
        noise, distance = rat_samples
        my_pos = board.player_worker.get_location()

        self.check_rat_resets(board)
        self.hmm.predict()
        self.hmm.update(noise, distance, my_pos, board)

        best_rat_loc, rat_prob = self.hmm.get_best_guess()

        turns_remaining = board.player_worker.turns_left
        time_left = time_left_func()
        
        # [REFACTOR] Smarter Time Management. 
        # Instead of allocating an exact fraction of total remaining time, we allow slightly 
        # more time per turn early on when board complexity is highest, while maintaining the buffer.
        base_time = (time_left / max(1, turns_remaining)) 
        time_limit = base_time - self.TIME_BUFFER
        if turns_remaining > 20: 
            time_limit = min(base_time * 1.5, time_left * 0.1) - self.TIME_BUFFER
            
        valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not valid_moves:
            return Move.search(best_rat_loc)

        carpet_moves = [m for m in valid_moves if m.move_type == MoveType.CARPET]
        if carpet_moves:
            best_carpet = max(carpet_moves, key=lambda m: m.roll_length)
            if best_carpet.roll_length >= 5: 
                return best_carpet

        start_time = time.time()
        best_tactical_move = random.choice(valid_moves)
        depth = 1
        
        # [REFACTOR] Dynamic iterative deepening exit. 
        # Instead of a flat 50% cutoff, we scale the cutoff based on how deep we are.
        # Shallow depths complete fast, so we keep going. Deep depths take exponentially 
        # longer, so we exit earlier to ensure we don't timeout mid-calculation.
        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > time_limit * (0.8 if depth < 3 else 0.4):
                    break
                    
                score, move = self.minimax(board, depth, float('-inf'), float('inf'), True, start_time, time_limit)
                if move is not None:
                    best_tactical_move = move
                depth += 1
        except TimeoutError:
            pass 

        adjusted_threshold = self.weights.get("rat_confidence_threshold", 0.65)
        if rat_prob >= adjusted_threshold: 
            return Move.search(best_rat_loc)

        return best_tactical_move

    def commentate(self):
        return "History Heuristic Online"