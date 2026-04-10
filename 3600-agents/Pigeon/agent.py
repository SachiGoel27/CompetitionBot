import numpy as np
import time
from game.enums import Cell, Noise, MoveType, Direction
from game.move import Move

NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE: (0.7, 0.15, 0.15),
    Cell.PRIMED: (0.1, 0.8, 0.1),
    Cell.CARPET: (0.1, 0.1, 0.8),
}

class RatBelief:
    def __init__(self, T_matrix):
        self.T = np.array(T_matrix)
        self.belief = np.zeros(64)
        self.reset_and_headstart()

    def reset_and_headstart(self):
        self.belief = np.zeros(64)
        self.belief[0] = 1.0
        T_1000 = np.linalg.matrix_power(self.T, 1000)
        self.belief = self.belief @ T_1000

    def predict(self):
        self.belief = self.belief @ self.T

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
        self.TIME_BUFFER = 0.1 

    def heuristic(self, board):
        score = board.player_worker.get_points() - board.opponent_worker.get_points()
        
        # Carpet Maximization: Non-linear scaling for contiguous primes
        prime_bonus = 0
        
        # Check rows for lines
        for y in range(8):
            row_chain = 0
            for x in range(8):
                if board.get_cell((x, y)) == Cell.PRIMED:
                    row_chain += 1
                    # Exponent of 1.5 rewards long lines heavily
                    prime_bonus += (row_chain ** 1.5) * 0.4
                else:
                    row_chain = 0
                    
        # Check columns for lines
        for x in range(8):
            col_chain = 0
            for y in range(8):
                if board.get_cell((x, y)) == Cell.PRIMED:
                    col_chain += 1
                    prime_bonus += (col_chain ** 1.5) * 0.4
                else:
                    col_chain = 0
                        
        return score + prime_bonus

    def minimax(self, board, depth, alpha, beta, is_maximizing, start_time, time_limit):
        if time.time() - start_time > time_limit:
            raise TimeoutError()

        if depth == 0 or board.is_game_over():
            return self.heuristic(board), None

        if is_maximizing:
            # OUR TURN
            valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
            if not valid_moves:
                return self.heuristic(board), None

            # Move Ordering
            def move_priority(m):
                if m.move_type == MoveType.CARPET: return 100 + m.roll_length
                if m.move_type == MoveType.PRIME: return 50
                return 0
                
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
                if beta <= alpha: break
            return max_eval, best_move
            
        else:
            # ENEMY TURN: We MUST reverse perspective before simulating!
            board.reverse_perspective()
            
            # Because we reversed, the enemy is now 'us' (enemy=False)
            valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
            
            if not valid_moves:
                board.reverse_perspective() # Revert before returning
                return self.heuristic(board), None

            best_move = valid_moves[0]
            min_eval = float('inf')
            
            for move in valid_moves:
                next_board = board.forecast_move(move, check_ok=False)
                if next_board is None: continue
                
                # next_board is currently from the enemy's perspective. Reverse it back to OURS!
                next_board.reverse_perspective()
                
                eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, True, start_time, time_limit)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    
                beta = min(beta, eval_score)
                if beta <= alpha: break
                
            # Revert the main board back to our perspective before the loop finishes
            board.reverse_perspective() 
            return min_eval, best_move

    def check_rat_resets(self, board):
        # Did we guess correctly?
        my_guess, my_correct = board.player_search
        if my_guess is not None and my_correct:
            self.hmm.reset_and_headstart()
            return
            
        # HMM FREE-RIDING: Exploit opponent's guesses
        op_guess, op_correct = board.opponent_search
        if op_guess is not None:
            if op_correct:
                self.hmm.reset_and_headstart()
            else:
                # They guessed wrong. We know 100% the rat is not there.
                op_idx = op_guess[1] * 8 + op_guess[0]
                self.hmm.belief[op_idx] = 0.0
                self.hmm.normalize()

    def play(self, board, rat_samples, time_left_func):
        noise, distance = rat_samples
        my_pos = board.player_worker.get_location()

        self.check_rat_resets(board)
        self.hmm.predict()
        self.hmm.update(noise, distance, my_pos, board)

        best_rat_loc, rat_prob = self.hmm.get_best_guess()
        ev_search = (rat_prob * 4) + ((1 - rat_prob) * -2)

        turns_remaining = board.player_worker.turns_left
        time_left = time_left_func()
        
        if turns_remaining > 0:
            time_limit = (time_left / turns_remaining) - self.TIME_BUFFER
        else:
            time_limit = time_left - self.TIME_BUFFER
            
        start_time = time.time()
        
        best_tactical_move = Move.plain(Direction.UP)
        best_tactical_score = float('-inf')

        depth = 1
        try:
            while True:
                # Stop deepening if we are halfway through our time budget
                if time.time() - start_time > time_limit * 0.5:
                    break
                    
                score, move = self.minimax(board, depth, float('-inf'), float('inf'), True, start_time, time_limit)
                if move is not None:
                    best_tactical_score = score
                    best_tactical_move = move
                depth += 1
                
        except TimeoutError:
            pass 

        current_board_score = self.heuristic(board)
        move_delta = best_tactical_score - current_board_score
        
        if ev_search > move_delta and ev_search > 0:
            return Move.search(best_rat_loc)

        return best_tactical_move

    def commentate(self):
        return "Lets get lit!"