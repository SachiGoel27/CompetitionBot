import numpy as np
import time
from game.enums import Cell, Noise, MoveType, Direction
from game.move import Move

# Noise probabilities extracted from rat.py
# Format: {CellType: (P(Squeak), P(Scratch), P(Squeal))}
NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE: (0.7, 0.15, 0.15),
    Cell.PRIMED: (0.1, 0.8, 0.1),
    Cell.CARPET: (0.1, 0.1, 0.8),
}

class RatBelief:
    def __init__(self, T_matrix):
        # The game engine provides T as a 2D array, representing P(from_index -> to_index)
        self.T = np.array(T_matrix)
        self.belief = np.zeros(64)
        self.reset_and_headstart()

    def reset_and_headstart(self):
        """Resets rat to (0,0) and simulates the 1000-move headstart."""
        self.belief = np.zeros(64)
        self.belief[0] = 1.0
        
        # Matrix power is efficient for simulating Markov Chain convergence
        T_1000 = np.linalg.matrix_power(self.T, 1000)
        self.belief = self.belief @ T_1000

    def predict(self):
        """Pushes the belief distribution forward by one turn."""
        self.belief = self.belief @ self.T

    def update(self, noise, reported_distance, my_pos, board):
        """Updates belief using Bayesian inference based on observations."""
        likelihood = np.zeros(64)
        
        my_x, my_y = my_pos
        
        for i in range(64):
            x, y = i % 8, i // 8
            cell_type = board.get_cell((x, y))
            
            # 1. Noise Likelihood
            p_noise = NOISE_PROBS[cell_type][noise.value]
            
            # 2. Distance Likelihood
            actual_dist = abs(my_x - x) + abs(my_y - y)
            p_dist = 0.0
            
            if reported_distance == 0:
                # Handle edge case where distance floor is 0
                if actual_dist == 0:
                    p_dist = 0.70 + 0.12 # Actual 0, offset 0 OR actual 0, offset -1 (floored)
                elif actual_dist == 1:
                    p_dist = 0.12 # Actual 1, offset -1 (floored to 0)
            else:
                if actual_dist == reported_distance:
                    p_dist = 0.70
                elif actual_dist == reported_distance + 1:
                    p_dist = 0.12  # Offset was -1
                elif actual_dist == reported_distance - 1:
                    p_dist = 0.12  # Offset was +1
                elif actual_dist == reported_distance - 2:
                    p_dist = 0.06  # Offset was +2
            
            likelihood[i] = p_noise * p_dist
            
        self.belief *= likelihood
        
        # Normalize
        total_prob = np.sum(self.belief)
        if total_prob > 0:
            self.belief /= total_prob
        else:
            # If numerical underflow occurs, reset to uniform distribution
            self.belief = np.ones(64) / 64.0

    def get_best_guess(self):
        """Returns the most probable location and its probability."""
        best_idx = np.argmax(self.belief)
        return (best_idx % 8, best_idx // 8), self.belief[best_idx]


class PlayerAgent:
    def __init__(self, board, transition_matrix, time_left_func):
        self.hmm = RatBelief(transition_matrix)
        
        # Time management constants
        self.TOTAL_TURNS = 40
        self.TIME_BUFFER = 0.1 # Leave 100ms safety buffer per move

    def heuristic(self, board):
        """
        Evaluates the board state. Rewards large point differentials
        and potential multi-square carpet setups.
        """
        score = board.player_worker.get_points() - board.opponent_worker.get_points()
        
        # Evaluate Prime Potential (Long horizontal/vertical lines of primes)
        my_pos = board.player_worker.get_location()
        enemy_pos = board.opponent_worker.get_location()
        
        prime_bonus = 0
        for y in range(8):
            for x in range(8):
                if board.get_cell((x, y)) == Cell.PRIMED:
                    # Minor bonus just for existing
                    prime_bonus += 0.5 
                    
                    # Proximity bonus - closer to me is better, closer to enemy is worse
                    my_dist = abs(my_pos[0] - x) + abs(my_pos[1] - y)
                    en_dist = abs(enemy_pos[0] - x) + abs(enemy_pos[1] - y)
                    if my_dist < en_dist:
                        prime_bonus += 1.0
                    else:
                        prime_bonus -= 1.0
                        
        return score + prime_bonus

    def minimax(self, board, depth, alpha, beta, is_maximizing, start_time, time_limit):
        """Alpha-Beta Minimax search tree."""
        # Time check - if we are out of time for this move, abort search
        if time.time() - start_time > time_limit:
            raise TimeoutError("Out of time budget")

        if depth == 0 or board.is_game_over():
            return self.heuristic(board), None

        # Exclude SEARCH moves from the deep tree to save branching factor.
        # We handle SEARCH mathematically at the root.
        valid_moves = board.get_valid_moves(enemy=not is_maximizing, exclude_search=True)
        
        if not valid_moves:
            return self.heuristic(board), None

        best_move = valid_moves[0]

        if is_maximizing:
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
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                next_board = board.forecast_move(move, check_ok=False)
                if next_board is None: continue
                
                # Note: Next board is evaluated from OUR perspective still
                eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, True, start_time, time_limit)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def check_rat_resets(self, board):
        """Determine if the rat was caught by either player to reset the HMM."""
        # Check if we caught it last turn
        if board.player_search[1] is True:
            self.hmm.reset_and_headstart()
            return
            
        # Check if opponent caught it last turn
        if board.opponent_search[1] is True:
            self.hmm.reset_and_headstart()
            return

    def play(self, board, rat_samples, time_left_func):
        noise, distance = rat_samples
        my_pos = board.player_worker.get_location()

        # 1. Update HMM
        self.check_rat_resets(board)
        self.hmm.predict()
        self.hmm.update(noise, distance, my_pos, board)

        # 2. Evaluate Search (Rat Catching) Expected Value
        best_rat_loc, rat_prob = self.hmm.get_best_guess()
        ev_search = (rat_prob * 4) + ((1 - rat_prob) * -2)

        # 3. Time Management for Iterative Deepening
        turns_remaining = board.player_worker.turns_left
        time_left = time_left_func()
        
        # Calculate strict time budget for this specific turn
        if turns_remaining > 0:
            time_limit = (time_left / turns_remaining) - self.TIME_BUFFER
        else:
            time_limit = time_left - self.TIME_BUFFER
            
        start_time = time.time()
        
        best_tactical_move = Move.plain(Direction.UP) # Fallback
        best_tactical_score = float('-inf')

        # 4. Iterative Deepening Minimax
        depth = 1
        try:
            while True:
                # Add check to prevent wasting time if we're near the time limit before starting a deep layer
                if time.time() - start_time > time_limit * 0.5:
                    break
                    
                score, move = self.minimax(board, depth, float('-inf'), float('inf'), True, start_time, time_limit)
                if move is not None:
                    best_tactical_score = score
                    best_tactical_move = move
                depth += 1
                
        except TimeoutError:
            # We hit our budget, use the best move from the last completed depth
            pass 

        # 5. Final Decision: Tactical vs. Search
        # We compare the tactical score to the expected value of searching.
        # Since 'best_tactical_score' includes total board points, we isolate the 
        # delta created by the move roughly by subtracting the current board heuristic.
        current_board_score = self.heuristic(board)
        move_delta = best_tactical_score - current_board_score
        
        # If the expected value of grabbing the rat is higher than the tactical advantage of moving
        if ev_search > move_delta and ev_search > 0:
            return Move.search(best_rat_loc)

        return best_tactical_move

    def commentate(self):
        return "I have tracked the rat, computed the odds, and rolled the carpet to victory."