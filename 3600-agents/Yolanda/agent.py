import random
from typing import Tuple
from collections.abc import Callable

from game import board, move, enums

class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        pass
        
    def commentate(self):
        return "I am Aggro-Yolanda, and I am here to carpet the world."

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        moves = board.get_valid_moves(exclude_search=True)
        
        best_move = moves[0]
        max_val = -999
        
        # Greedy logic: Assign immediate values to moves
        for m in moves:
            val = 0
            if m.move_type == enums.MoveType.CARPET:
                val = 100 + m.roll_length  # Always take the longest carpet possible
            elif m.move_type == enums.MoveType.PRIME:
                val = 50  # Always prime if you can't carpet
            elif m.move_type == enums.MoveType.PLAIN:
                val = 10  # Move plain if stuck
            
            # Add a tiny bit of random noise so she doesn't get stuck in infinite loops
            val += random.uniform(0, 5)
            
            if val > max_val:
                max_val = val
                best_move = m
                
        return best_move