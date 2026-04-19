import numpy as np
import time
import os
import json
import random
from collections import defaultdict
from game.enums import Cell, Noise, MoveType, Direction, CARPET_POINTS_TABLE
from game.move import Move

# [FIXED] Unified Static Dictionary based on telemetry
NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE: (0.48, 0.26, 0.26),
    Cell.PRIMED: (0.48, 0.26, 0.26),
    Cell.CARPET: (0.48, 0.26, 0.26),
}

# ==========================================
# ZOBRIST HASHING INITIALIZATION
# ==========================================
ZOBRIST_TABLE = {}
for x in range(8):
    for y in range(8):
        for cell_type in Cell:
            ZOBRIST_TABLE[(x, y, cell_type)] = random.getrandbits(64)

ZOBRIST_TURN_MAX = random.getrandbits(64)

# ==========================================
# THE TRACKER (HMM) - VECTORIZED
# ==========================================
class RatBelief:
    def __init__(self, T_matrix):
        self.T = np.array(T_matrix)
        
        self.grid_x = np.arange(64) % 8
        self.grid_y = np.arange(64) // 8
        
        initial_state = np.zeros(64)
        initial_state[0] = 1.0
        T_1000 = np.linalg.matrix_power(self.T, 1000) 
        self.stationary_belief = initial_state @ T_1000
        
        self.belief = np.zeros(64)
        self.reset_and_headstart(is_initial_spawn=True)

    def reset_and_headstart(self, is_initial_spawn=False, board=None):
        if is_initial_spawn:
            self.belief = self.stationary_belief.copy()
        else:
            self.belief = np.zeros(64)
            valid_cells = 0
            
            for i in range(64):
                x, y = i % 8, i // 8
                if board.get_cell((x, y)) != Cell.BLOCKED:
                    self.belief[i] = 1.0
                    valid_cells += 1
                    
            if valid_cells > 0:
                self.belief /= valid_cells
            else:
                self.belief = np.ones(64) / 64.0

    def predict(self):
        self.belief = self.belief @ self.T 

    def update(self, noise, reported_distance, my_pos, board):
        my_x, my_y = my_pos
        
        p_noises = np.array([
            NOISE_PROBS[board.get_cell((i % 8, i // 8))][noise.value] 
            for i in range(64)
        ])
        
        actual_dists = np.abs(my_x - self.grid_x) + np.abs(my_y - self.grid_y)
        p_dists = np.zeros(64)
        
        if reported_distance == 0:
            p_dists[actual_dists == 0] = 0.82
            p_dists[actual_dists == 1] = 0.12
        else:
            p_dists[actual_dists == reported_distance] = 0.70
            p_dists[actual_dists == reported_distance + 1] = 0.12
            p_dists[actual_dists == reported_distance - 1] = 0.12
            p_dists[actual_dists == reported_distance - 2] = 0.06
            
        likelihood = p_noises * p_dists
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


# ==========================================
# THE AI AGENT (MINIMAX + PHASE EVAL)
# ==========================================
class PlayerAgent:
    def __init__(self, board, transition_matrix, time_left_func):
        self.hmm = RatBelief(transition_matrix)
        self.TIME_BUFFER = 0.8 
        self.tt = {} 
        self.history_table = defaultdict(int) 
        self.last_search_depth = 0
        
        self.metrics = {
            "total_searches": 0,
            "ghost_rats": 0,
            "max_false_confidence": 0.0
        }
        self.last_search_confidence = 0.0
        self.noise_tally = {
            Cell.SPACE: {0: 0, 1: 0, 2: 0},
            Cell.PRIMED: {0: 0, 1: 0, 2: 0}
        }
        
        self.weights = {
            "rat_confidence_threshold": 0.65
        }

    def check_rat_resets(self, board):
        my_guess, my_correct = board.player_search
        if my_guess is not None:
            self.metrics["total_searches"] += 1
            if my_correct:
                self.hmm.reset_and_headstart(is_initial_spawn=False, board=board)
                self.last_search_confidence = 0.0 
            else:
                if self.last_search_confidence >= self.weights["rat_confidence_threshold"]:
                    self.metrics["ghost_rats"] += 1
                    self.metrics["max_false_confidence"] = max(self.metrics["max_false_confidence"], self.last_search_confidence)
                
                my_idx = my_guess[1] * 8 + my_guess[0]
                self.hmm.belief[my_idx] = 0.0
                self.hmm.normalize()
                self.last_search_confidence = 0.0 
                
        op_guess, op_correct = board.opponent_search
        if op_guess is not None:
            if op_correct:
                self.hmm.reset_and_headstart(is_initial_spawn=False, board=board)
            else:
                op_idx = op_guess[1] * 8 + op_guess[0]
                self.hmm.belief[op_idx] = 0.0
                self.hmm.normalize()

    def _hash_board(self, board, is_maximizing):
        h = 0
        for x in range(8):
            for y in range(8):
                cell = board.get_cell((x, y))
                h ^= ZOBRIST_TABLE[(x, y, cell)]
                
        p_x, p_y = board.player_worker.get_location()
        e_x, e_y = board.opponent_worker.get_location()
        
        h ^= (p_x * 73856093 ^ p_y * 19349663)
        h ^= (e_x * 83492791 ^ e_y * 39916801)
        h ^= (board.player_worker.get_points() * 104729)
        h ^= (board.opponent_worker.get_points() * 104723)
        
        if is_maximizing:
            h ^= ZOBRIST_TURN_MAX
            
        return h
    
    def heuristic(self, board, turns_left):
        score = board.player_worker.get_points() - board.opponent_worker.get_points()
        my_x, my_y = board.player_worker.get_location()
        en_x, en_y = board.opponent_worker.get_location()
        
        def quick_dist(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)

        def evaluate_line(start_x, start_y, end_x, end_y, length, is_horizontal):
            if length == 0: return 0
            
            actual_points = CARPET_POINTS_TABLE[min(length, 7)] 
            potential_value = actual_points * 0.7 

            if turns_left <= 15:
                depreciation_multiplier = max(0.2, turns_left / 15.0)
                potential_value *= depreciation_multiplier
            
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
            
            # [FIXED] Greedy Selfish Evaluation
            if my_dist == 0:
                multiplier = 1.5
            elif my_dist < en_dist:
                multiplier = 1.0 + (en_dist - my_dist) * 0.1
            elif my_dist == en_dist:
                multiplier = 0.5
            else:
                # The enemy is winning the race. The line is dead to us.
                multiplier = 0.0
                
            return potential_value * multiplier

        prime_bonus = 0.0

        for y in range(8):
            length, s_x, s_y = 0, 0, y
            for x in range(8):
                if board.get_cell((x, y)) == Cell.PRIMED:
                    if length == 0: s_x = x
                    length += 1
                elif length > 0:
                    prime_bonus += evaluate_line(s_x, s_y, x - 1, y, length, True)
                    length = 0
            if length > 0: prime_bonus += evaluate_line(s_x, s_y, 7, y, length, True)
                
        for x in range(8):
            length, s_x, s_y = 0, x, 0
            for y in range(8):
                if board.get_cell((x, y)) == Cell.PRIMED:
                    if length == 0: s_y = y
                    length += 1
                elif length > 0:
                    prime_bonus += evaluate_line(s_x, s_y, x, y - 1, length, False)
                    length = 0
            if length > 0: prime_bonus += evaluate_line(s_x, s_y, x, 7, length, False)
            
        center_bias = (abs(3.5 - my_x) + abs(3.5 - my_y)) * -0.01 
        
        return score + (prime_bonus * 0.8) + center_bias

    def quiescence_search(self, board, alpha, beta, is_maximizing, start_time, time_limit, turns_left):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
            
        stand_pat = self.heuristic(board, turns_left)
        
        if is_maximizing:
            if stand_pat >= beta: return beta
            if alpha < stand_pat: alpha = stand_pat
            
            valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
            tactical_moves = [m for m in valid_moves if m.move_type == MoveType.CARPET]
            tactical_moves.sort(key=lambda m: getattr(m, 'roll_length', 0), reverse=True)
            
            if not tactical_moves: return stand_pat
                
            for move in tactical_moves:
                next_board = board.forecast_move(move, check_ok=False)
                if next_board is None: continue
                
                score = self.quiescence_search(next_board, alpha, beta, False, start_time, time_limit, turns_left - 1)
                
                if score >= beta: return beta
                if score > alpha: alpha = score
            return alpha
            
        else:
            if stand_pat <= alpha: return alpha
            if beta > stand_pat: beta = stand_pat

            board.reverse_perspective()
            valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
            tactical_moves = [m for m in valid_moves if m.move_type == MoveType.CARPET]
            tactical_moves.sort(key=lambda m: getattr(m, 'roll_length', 0), reverse=True)
            
            if not tactical_moves:
                board.reverse_perspective()
                return stand_pat
                
            for move in tactical_moves:
                next_board = board.forecast_move(move, check_ok=False)
                if next_board is None: continue
                
                next_board.reverse_perspective()
                score = self.quiescence_search(next_board, alpha, beta, True, start_time, time_limit, turns_left - 1)
                
                if score <= alpha: 
                    board.reverse_perspective()
                    return alpha
                if score < beta: beta = score
                
            board.reverse_perspective()
            return beta

    def minimax(self, board, depth, alpha, beta, is_maximizing, start_time, time_limit, turns_left):
        if time.time() - start_time > time_limit:
            raise TimeoutError()

        if board.is_game_over():
            return self.heuristic(board, turns_left), None
            
        if depth == 0:
            return self.quiescence_search(board, alpha, beta, is_maximizing, start_time, time_limit, turns_left), None

        state_hash = self._hash_board(board, is_maximizing)
        hash_move = None
        
        if state_hash in self.tt:
            cached_depth, cached_score, cached_move, flag = self.tt[state_hash]
            hash_move = cached_move
            if cached_depth >= depth:
                if flag == 'EXACT': return cached_score, cached_move
                elif flag == 'LOWERBOUND': alpha = max(alpha, cached_score)
                elif flag == 'UPPERBOUND': beta = min(beta, cached_score)
                if alpha >= beta: return cached_score, cached_move

        orig_alpha = alpha
        orig_beta = beta

        if is_maximizing:
            valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
            if not valid_moves: return self.heuristic(board, turns_left), None

            def move_priority(m):
                if m == hash_move: return float('inf') 
                base = self.history_table.get(str(m), 0) 
                if m.move_type == MoveType.CARPET: return base + 100 + (m.roll_length * 2)
                if m.move_type == MoveType.PRIME: return base + 50
                return base
                
            valid_moves.sort(key=move_priority, reverse=True)
            best_move = valid_moves[0]
            max_eval = float('-inf')
            
            bSearchPv = True 
            for move in valid_moves:
                next_board = board.forecast_move(move, check_ok=False)
                if next_board is None: continue
                
                if bSearchPv:
                    eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, False, start_time, time_limit, turns_left - 1)
                else:
                    eval_score, _ = self.minimax(next_board, depth - 1, alpha, alpha + 1, False, start_time, time_limit, turns_left - 1)
                    if alpha < eval_score < beta: 
                        eval_score, _ = self.minimax(next_board, depth - 1, eval_score, beta, False, start_time, time_limit, turns_left - 1)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha: 
                    self.history_table[str(move)] += (depth ** 2) 
                    break
                bSearchPv = False
                
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
                return self.heuristic(board, turns_left), None

            def min_move_priority(m):
                if m == hash_move: return float('inf')
                return self.history_table.get(str(m), 0)

            valid_moves.sort(key=min_move_priority, reverse=True)
            best_move = valid_moves[0]
            min_eval = float('inf')
            
            bSearchPv = True
            for move in valid_moves:
                next_board = board.forecast_move(move, check_ok=False)
                if next_board is None: continue
                
                next_board.reverse_perspective()
                
                if bSearchPv:
                    eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, True, start_time, time_limit, turns_left - 1)
                else:
                    eval_score, _ = self.minimax(next_board, depth - 1, beta - 1, beta, True, start_time, time_limit, turns_left - 1)
                    if alpha < eval_score < beta:
                        eval_score, _ = self.minimax(next_board, depth - 1, alpha, eval_score, True, start_time, time_limit, turns_left - 1)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    
                beta = min(beta, eval_score)
                if beta <= alpha: 
                    self.history_table[str(move)] += (depth ** 2)
                    break
                bSearchPv = False
                
            board.reverse_perspective() 
            flag = 'EXACT'
            if min_eval <= alpha: flag = 'UPPERBOUND'
            elif min_eval >= orig_beta: flag = 'LOWERBOUND'
            self.tt[state_hash] = (depth, min_eval, best_move, flag)
            return min_eval, best_move

    def play(self, board, rat_samples, time_left_func):
        noise, distance = rat_samples
        my_pos = board.player_worker.get_location()
        turns_remaining = board.player_worker.turns_left

        # [NEW] Raw Sensor Override: Strike instantly if the rat is highly likely here.
        if distance == 0:
            return Move.search(my_pos)

        current_cell = board.get_cell(my_pos)
        if current_cell in self.noise_tally:
            self.noise_tally[current_cell][noise.value] += 1

        self.check_rat_resets(board)
        self.hmm.predict()
        self.hmm.update(noise, distance, my_pos, board)

        best_rat_loc, rat_prob = self.hmm.get_best_guess()

        time_left = time_left_func()
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
        
        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > time_limit * (0.8 if depth < 3 else 0.4):
                    break
                    
                score, move = self.minimax(board, depth, float('-inf'), float('inf'), True, start_time, time_limit, turns_remaining)
                if move is not None:
                    best_tactical_move = move
                depth += 1
        except TimeoutError:
            pass 
            
        self.last_search_depth = depth - 1
        
        adjusted_threshold = self.weights.get("rat_confidence_threshold", 0.65)
        if rat_prob >= adjusted_threshold: 
            self.last_search_confidence = rat_prob
            return Move.search(best_rat_loc)

        return best_tactical_move

    def commentate(self):
        searches = self.metrics["total_searches"]
        ghosts = self.metrics["ghost_rats"]
        max_conf = self.metrics["max_false_confidence"]
        
        ghost_rate = (ghosts / searches * 100) if searches > 0 else 0.0
        base_log = f"Depth: {self.last_search_depth} | Ghost Rats: {ghosts} ({ghost_rate:.1f}%) | Max False Conf: {max_conf:.2f}"
        
        if hasattr(self, 'noise_tally'):
            telemetry = []
            for c_type in [Cell.SPACE, Cell.PRIMED]:
                counts = self.noise_tally[c_type]
                total = sum(counts.values())
                if total > 0:
                    p0 = counts[0] / total
                    p1 = counts[1] / total
                    p2 = counts[2] / total
                    telemetry.append(f"{c_type.name}: ({p0:.3f}, {p1:.3f}, {p2:.3f}) [n={total}]")
            
            if telemetry:
                return base_log + " || " + " | ".join(telemetry)
                
        return base_log