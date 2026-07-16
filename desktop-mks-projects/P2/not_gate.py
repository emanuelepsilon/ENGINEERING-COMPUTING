import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# Define constants for the cell states to improve readability.
ALIVE_CELL = 255
DEAD_CELL = 0
MAX_ROWS = 40
MAX_COLS = 80

# ---------------------------------------------------------
# PATTERN GENERATORS
# These functions define the local coordinate configurations 
# for specific Game of Life structures (gliders, eaters).
# The row_offset and col_offset parameters allow us to place
# these structures anywhere in the cellular universe.
# ---------------------------------------------------------

def create_forward_glider(row_offset, col_offset):
    # A standard glider pattern that moves diagonally.
    glider_coords = [[5, 1],[6, 1],[6, 2],[5, 2],[5, 5],[4, 6],[5, 6],[6, 6],[3, 7],[7, 7],[5, 8],[2, 9],[2, 10],
                     [3, 11], [4, 12],[5, 12],[6, 12],[7, 11],[8, 10],[8, 9],[7, 17],[7, 20],[7, 21],[7, 22],[6, 22],
                     [5, 21],[6, 26],[5, 26],[1, 26],[0, 26],[1, 28],[5, 28],[2, 29],[3, 29],[4, 29],[4, 30],[3, 30],
                     [2, 30],[3, 35],[4, 35],[4, 36],[3, 36]]
    for cell_coord in glider_coords:
        cell_coord[0] += row_offset
        cell_coord[1] += col_offset
    return glider_coords

def create_backward_glider(row_offset, col_offset):
    # A reverse glider pattern that travels in the opposite diagonal direction.
    reverse_glider_coords = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 6], [1, 6], [1, 7], [0, 7], [-1, 7], [-1, 6], [-2, 8], 
                             [-2, 10], [-3, 10], [2, 8], [2, 10], [3, 10], [3, 14], [4, 14], [4, 15], [4, 16], [2, 15], 
                             [4, 19], [4, 25], [3, 24], [2, 24], [1, 24], [0, 25], [-1, 26], [-1, 27], [5, 26], [5, 27], 
                             [2, 28], [2, 30], [2, 31], [1, 30], [3, 30], [0, 29], [4, 29], [2, 34], [2, 35], [3, 35], [3, 34]]
    for cell_coord in reverse_glider_coords:
        cell_coord[0] += row_offset
        cell_coord[1] += col_offset
    return reverse_glider_coords

def create_eater_not():
    # A stable "eater" structure specific to the NOT gate that consumes incoming gliders.
    return [[32, 24], [32, 25], [33, 24], [33, 24], [33, 25], [32, 28], [32, 29], [33, 28],
            [34, 29], [34, 30], [34, 31], [34, 32], [33, 32], [36, 31], [36, 32], [37, 32], [37, 31]]

def create_stopper_not():
    # A block structure used to intercept and stop the signal when the input is 0.
    return [[14, 27], [15, 27], [14, 28], [16, 28], [16, 29], [16, 30], [17, 30]]

# ---------------------------------------------------------
# GAME OF LIFE ENGINE
# These functions handle the mathematical simulation of the grid.
# ---------------------------------------------------------

def count_living_neighbors(universe_state, row_idx, col_idx):
    # This function scans the 3x3 perimeter around a specific cell coordinates.
    # It counts how many of those adjacent 8 cells are currently ALIVE.
    total_alive = 0
    for r in range(row_idx - 1, row_idx + 2):
        for c in range(col_idx - 1, col_idx + 2):
            try:
                # Add to total if the neighbor is alive.
                if universe_state[r, c] == ALIVE_CELL:
                    total_alive += 1
            except IndexError:
                # Ignore edges/out of bounds queries.
                pass
    
    # We scanned the 3x3 block which includes the core cell itself.
    # If the core cell is alive, we must subtract it from the neighbor count.
    if universe_state[row_idx, col_idx] == ALIVE_CELL:
        total_alive -= 1
        
    return total_alive

def compute_next_generation(frame_data):
    # This function is called every frame to apply Conway's rules.
    global cellular_universe, animation_matrix
    
    # We must evaluate the rules simultaneously, so we read from the current universe
    # but write the results to a completely new grid.
    next_generation_universe = cellular_universe.copy()
    
    for row_idx in range(MAX_ROWS):
        for col_idx in range(MAX_COLS):
            total_alive = count_living_neighbors(cellular_universe, row_idx, col_idx)
            
            # Rule 1 & 2: A live cell dies if it has fewer than 2 (loneliness) or more than 3 (overpopulation) neighbors.
            if cellular_universe[row_idx, col_idx] == ALIVE_CELL:
                if (total_alive < 2) or (total_alive > 3):
                    next_generation_universe[row_idx, col_idx] = DEAD_CELL
            # Rule 4: A dead cell comes to life if it has exactly 3 neighbors (reproduction).
            else:
                if total_alive == 3:
                    next_generation_universe[row_idx, col_idx] = ALIVE_CELL
                    
    # Update the visual matrix with the new state
    animation_matrix.set_data(next_generation_universe)
    cellular_universe = next_generation_universe
    return [animation_matrix]

# ---------------------------------------------------------
# MAIN EXECUTION
# Initializes the logic gate based on terminal arguments.
# ---------------------------------------------------------
if __name__ == '__main__':
    # Validate user inputs
    if len(sys.argv) < 2:
        print("Usage: python not_gate.py <input_A>")
        sys.exit()

    input_A = int(sys.argv[1])
    coordinates_to_activate = []
    
    # Always construct the base components of the NOT gate.
    coordinates_to_activate.append(create_forward_glider(1, 0))
    coordinates_to_activate.append(create_backward_glider(3, 40))
    coordinates_to_activate.append(create_eater_not())

    # Apply the NOT logic: If the input is 0, we spawn a stopper to intercept the signal.
    if input_A == 0:
        coordinates_to_activate.append(create_stopper_not())

    # Create the empty universe with correct dimensions.
    cellular_universe = np.zeros((MAX_ROWS, MAX_COLS))
    
    # Ignite the initial alive cells onto the grid based on our coordinates.
    for pattern_group in coordinates_to_activate:
        for cell_coord in pattern_group:
            cellular_universe[cell_coord[0], cell_coord[1]] = ALIVE_CELL

    # Set up the visualization with Matplotlib.
    fig, ax = plt.subplots()
    
    # Apply cmap='binary' to make dead cells = white and alive cells = black.
    animation_matrix = ax.matshow(cellular_universe, cmap='binary', vmin=DEAD_CELL, vmax=ALIVE_CELL)
    
    ani = animation.FuncAnimation(fig, compute_next_generation, interval=1, save_count=5)
    plt.title(f"NOT Gate (A={input_A})")
    plt.show()