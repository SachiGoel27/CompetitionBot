import json
import math
import os
import time
import numpy as np

from game.enums import Cell, Noise, MoveType, Direction
from game.move import Move

CARPET_POINTS = {
    1: -1,
    2: 2,
    3: 4,
    4: 6,
    5: 10,
    6: 15,
    7: 21,
}

NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE:   (0.7, 0.15, 0.15),
    Cell.PRIMED:  (0.1, 0.8, 0.1),
    Cell.CARPET:  (0.1, 0.1, 0.8),
}

DIRECTIONS = [
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
]

class RatBelief:
    def __init__(self, transition_matrix):
        self.T = np.array(transition_matrix, dtype=np.float64)
        self.belief = np.zeros(64, dtype=np.float64)
        self.reset_and_headstart()

    def reset_and_headstart(self):
        self.belief = np.zeros(64, dtype=np.float64)
        self.belief[0] = 1.0
        T_1000 = np.linalg.matrix_power(self.T, 1000)
        self.belief = self.belief @ T_1000
        self.normalize()

    def predict(self):
        self.belief = self.belief @ self.T
        self.normalize()

    def update(self, noise, reported_distance, my_pos, board):
        likelihood = np.zeros(64, dtype=np.float64)
        my_x, my_y = my_pos

        for i in range(64):
            x, y = i % 8, i // 8

            cell_type = board.get_cell((x, y))
            probs = NOISE_PROBS.get(cell_type, NOISE_PROBS[Cell.SPACE])

            p_noise = probs[noise.value]

            actual_dist = abs(my_x - x) + abs(my_y - y)
            p_dist = self._distance_likelihood(actual_dist, reported_distance)

            likelihood[i] = p_noise * p_dist

        self.belief *= likelihood
        self.normalize()

    def _distance_likelihood(self, actual_dist, reported_distance):

        probs = 0.0

        if reported_distance == actual_dist:
            probs += 0.70

        if max(0, actual_dist - 1) == reported_distance:
            probs += 0.12

        if actual_dist + 1 == reported_distance:
            probs += 0.12

        if actual_dist + 2 == reported_distance:
            probs += 0.06

        return probs

    def normalize(self):
        s = np.sum(self.belief)
        if s > 0:
            self.belief /= s
        else:
            self.belief[:] = 1.0 / 64.0

    def remove_impossible_location(self, pos):
        idx = pos[1] * 8 + pos[0]
        self.belief[idx] = 0.0
        self.normalize()

    def get_best_guess(self):
        idx = int(np.argmax(self.belief))
        return (idx % 8, idx // 8), float(self.belief[idx])

    def get_top_two(self):
        order = np.argsort(self.belief)
        best = int(order[-1])
        second = int(order[-2])
        best_pos = (best % 8, best // 8)
        second_pos = (second % 8, second // 8)
        return (
            best_pos, float(self.belief[best]),
            second_pos, float(self.belief[second])
        )

    def entropy(self):
        eps = 1e-12
        return float(-np.sum(self.belief * np.log(self.belief + eps)))

class PlayerAgent:
    def __init__(self, board, transition_matrix, time_left_func):
        self.hmm = RatBelief(transition_matrix)

        self.TIME_BUFFER = 0.08
        self.MIN_MOVE_BUDGET = 0.01

        self.weights = {
            "score_diff": 10.0,
            "immediate_gain": 3.4,
            "opp_immediate_gain": 2.8,
            "mobility": 0.28,
            "center_bonus": 0.85,
            "potential": 1.6,
            "distance_to_potential": 0.55,
            "carpet_presence": 0.3,
            "opp_carpet_presence": 0.2,
        }

        if "AGENT_WEIGHTS" in os.environ:
            try:
                self.weights.update(json.loads(os.environ["AGENT_WEIGHTS"]))
            except Exception:
                pass


    def in_bounds(self, x, y):
        return 0 <= x < 8 and 0 <= y < 8

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_cell_safe(self, board, pos):
        x, y = pos
        if not self.in_bounds(x, y):
            return None
        return board.get_cell(pos)

    def _get_valid_moves(self, board, enemy=False, exclude_search=True):
        if not enemy:
            try:
                return list(board.get_valid_moves(enemy=False, exclude_search=exclude_search))
            except Exception:
                return []
        else:
            board.reverse_perspective()
            try:
                return list(board.get_valid_moves(enemy=False, exclude_search=exclude_search))
            except Exception:
                return []
            finally:
                board.reverse_perspective()

    def _safe_fallback_move(self, board):
        moves = self._get_valid_moves(board, enemy=False, exclude_search=True)
        if not moves:
            return Move.plain(Direction.UP)
        moves.sort(key=lambda m: self.move_priority(board, m), reverse=True)
        return moves[0]

    def _move_points(self, move):
        move_type = getattr(move, "move_type", None)
        if move_type == MoveType.PRIME:
            return 1
        if move_type == MoveType.CARPET:
            k = int(getattr(move, "roll_length", 1))
            return CARPET_POINTS.get(k, 0)
        return 0

    def _count_cells(self, board, target_cell):
        total = 0
        for y in range(8):
            for x in range(8):
                if board.get_cell((x, y)) == target_cell:
                    total += 1
        return total

    def _center_bonus(self, pos):
        x, y = pos
        if 2 <= x <= 5 and 2 <= y <= 5:
            return 1.0
        if 1 <= x <= 6 and 1 <= y <= 6:
            return 0.35
        return 0.0

    def _mobility(self, board, enemy=False):
        return len(self._get_valid_moves(board, enemy=enemy, exclude_search=True))

    def _line_run_through_cell(self, board, pos, axis):
        x, y = pos
        if axis == "h":
            left = 0
            cx = x - 1
            while cx >= 0 and board.get_cell((cx, y)) == Cell.PRIMED:
                left += 1
                cx -= 1

            right = 0
            cx = x + 1
            while cx < 8 and board.get_cell((cx, y)) == Cell.PRIMED:
                right += 1
                cx += 1
            return left + right

        up = 0
        cy = y - 1
        while cy >= 0 and board.get_cell((x, cy)) == Cell.PRIMED:
            up += 1
            cy -= 1

        down = 0
        cy = y + 1
        while cy < 8 and board.get_cell((x, cy)) == Cell.PRIMED:
            down += 1
            cy += 1
        return up + down

    def _cell_potential(self, board, pos):
        cell = board.get_cell(pos)

        if cell == Cell.BLOCKED or cell == Cell.CARPET:
            return -999.0

        horiz = self._line_run_through_cell(board, pos, "h")
        vert = self._line_run_through_cell(board, pos, "v")

        adjacency = 0
        x, y = pos
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                if board.get_cell((nx, ny)) == Cell.PRIMED:
                    adjacency += 1

        return 1.4 * max(horiz, vert) + 0.8 * min(horiz, vert) + 0.6 * adjacency

    def _best_potential_value_and_distance(self, board, from_pos):
        best_value = -999.0
        best_dist = 20

        for y in range(8):
            for x in range(8):
                cell = board.get_cell((x, y))
                if cell == Cell.BLOCKED or cell == Cell.CARPET:
                    continue

                pot = self._cell_potential(board, (x, y))
                dist = self.manhattan(from_pos, (x, y))

                if pot > best_value or (pot == best_value and dist < best_dist):
                    best_value = pot
                    best_dist = dist

        return best_value, best_dist

    def _best_immediate_gain(self, board, enemy=False):
        moves = self._get_valid_moves(board, enemy=enemy, exclude_search=True)
        if not moves:
            return 0.0
        return max(self._move_points(m) for m in moves)

    def _estimate_board_value(self, board):
        my_score = board.player_worker.get_points()
        opp_score = board.opponent_worker.get_points()
        score_diff = my_score - opp_score

        my_pos = board.player_worker.get_location()
        opp_pos = board.opponent_worker.get_location()

        my_best_gain = self._best_immediate_gain(board, enemy=False)
        opp_best_gain = self._best_immediate_gain(board, enemy=True)

        my_mobility = self._mobility(board, enemy=False)
        opp_mobility = self._mobility(board, enemy=True)

        my_potential, my_potential_dist = self._best_potential_value_and_distance(board, my_pos)
        opp_potential, opp_potential_dist = self._best_potential_value_and_distance(board, opp_pos)

        my_carpet = self._count_cells(board, Cell.CARPET)
        my_primed = self._count_cells(board, Cell.PRIMED)

        value = 0.0
        value += self.weights["score_diff"] * score_diff
        value += self.weights["immediate_gain"] * my_best_gain
        value -= self.weights["opp_immediate_gain"] * opp_best_gain
        value += self.weights["mobility"] * (my_mobility - opp_mobility)
        value += self.weights["center_bonus"] * self._center_bonus(my_pos)
        value -= 0.6 * self._center_bonus(opp_pos)
        value += self.weights["potential"] * my_potential
        value -= 0.9 * opp_potential
        value -= self.weights["distance_to_potential"] * my_potential_dist
        value += 0.30 * opp_potential_dist
        value += self.weights["carpet_presence"] * my_carpet
        value += 0.12 * my_primed

        return value

    def move_priority(self, board, move):
        base = 0.0
        immediate = self._move_points(move)

        move_type = getattr(move, "move_type", None)
        if move_type == MoveType.CARPET:
            base += 100.0 + 18.0 * immediate + 2.0 * getattr(move, "roll_length", 1)
        elif move_type == MoveType.PRIME:
            base += 50.0 + 10.0 * immediate
        else:
            base += 5.0

        try:
            next_board = board.forecast_move(move, check_ok=False)
            if next_board is not None:
                base += 0.10 * self._estimate_board_value(next_board)
        except Exception:
            pass

        return base

    def minimax(self, board, depth, alpha, beta, is_maximizing, start_time, time_limit):
        if time.time() - start_time >= time_limit:
            raise TimeoutError()

        if depth == 0 or board.is_game_over():
            return self._estimate_board_value(board), None

        if is_maximizing:
            moves = self._get_valid_moves(board, enemy=False, exclude_search=True)
            if not moves:
                return self._estimate_board_value(board), None

            moves.sort(key=lambda m: self.move_priority(board, m), reverse=True)

            moves = moves[:8]

            best_move = moves[0]
            best_val = float("-inf")

            for move in moves:
                if time.time() - start_time >= time_limit:
                    raise TimeoutError()

                next_board = board.forecast_move(move, check_ok=False)
                if next_board is None:
                    continue

                val, _ = self.minimax(
                    next_board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    start_time,
                    time_limit
                )

                if val > best_val:
                    best_val = val
                    best_move = move

                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break

            return best_val, best_move

        board.reverse_perspective()
        try:
            moves = self._get_valid_moves(board, enemy=False, exclude_search=True)
            if not moves:
                return self._estimate_board_value(board), None

            moves.sort(key=lambda m: self.move_priority(board, m), reverse=True)
            moves = moves[:8]

            best_move = moves[0]
            best_val = float("inf")

            for move in moves:
                if time.time() - start_time >= time_limit:
                    raise TimeoutError()

                next_board = board.forecast_move(move, check_ok=False)
                if next_board is None:
                    continue

                next_board.reverse_perspective()

                val, _ = self.minimax(
                    next_board,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    start_time,
                    time_limit
                )

                if val < best_val:
                    best_val = val
                    best_move = move

                beta = min(beta, best_val)
                if beta <= alpha:
                    break

            return best_val, best_move
        finally:
            board.reverse_perspective()

    def check_rat_resets(self, board):
        my_guess, my_correct = board.player_search
        if my_guess is not None and my_correct:
            self.hmm.reset_and_headstart()
            return

        opp_guess, opp_correct = board.opponent_search
        if opp_guess is not None:
            if opp_correct:
                self.hmm.reset_and_headstart()
            else:
                self.hmm.remove_impossible_location(opp_guess)

    def should_search(self, board, top_prob, second_prob, tactical_gain):
        turns_left = board.player_worker.turns_left
        ev_search = 6.0 * top_prob - 2.0
        margin = top_prob - second_prob
        entropy = self.hmm.entropy()

        if turns_left > 24:
            return (
                top_prob >= 0.48 and
                margin >= 0.10 and
                ev_search > max(0.8, tactical_gain) and
                entropy < 2.7
            )

        if turns_left > 10:
            return (
                top_prob >= 0.42 and
                margin >= 0.07 and
                ev_search > (tactical_gain - 0.3) and
                entropy < 3.1
            )

        return (
            top_prob >= 0.36 and
            ev_search > (tactical_gain - 0.8)
        )

    def play(self, board, rat_samples, time_left_func):
        noise, distance = rat_samples

        self.check_rat_resets(board)

        my_pos = board.player_worker.get_location()

        self.hmm.predict()
        self.hmm.update(noise, distance, my_pos, board)

        best_rat_pos, best_rat_prob, second_rat_pos, second_rat_prob = self.hmm.get_top_two()

        fallback_move = self._safe_fallback_move(board)

        time_left = time_left_func()
        turns_left = max(1, board.player_worker.turns_left)

        per_turn_budget = max(
            self.MIN_MOVE_BUDGET,
            (time_left / turns_left) - self.TIME_BUFFER
        )

        start_time = time.time()

        best_move = fallback_move
        best_value = self._estimate_board_value(board)
        current_value = best_value

        depth = 1
        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed >= per_turn_budget * 0.85:
                    break

                value, move = self.minimax(
                    board,
                    depth,
                    float("-inf"),
                    float("inf"),
                    True,
                    start_time,
                    per_turn_budget
                )

                if move is not None:
                    best_move = move
                    best_value = value

                depth += 1

                if depth > 5:
                    break

        except TimeoutError:
            pass

        tactical_gain = best_value - current_value

        if self.should_search(board, best_rat_prob, second_rat_prob, tactical_gain):
            return Move.search(best_rat_pos)

        return best_move

    def commentate(self):
        return "Calculated chaos."