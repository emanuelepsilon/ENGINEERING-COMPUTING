import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# Define constants for the cell states
ALIVE_CELL = 255
DEAD_CELL = 0
MAX_ROWS = 80
MAX_COLS = 158

# ---------------------------------------------------------
# PATTERN GENERATORS
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

def create_eater_or():
    # Pattern designed to eliminate stray gliders at the end of the OR signal line.
    return [[72, 82], [73, 82], [73, 83], [73, 84], [73, 85], [72, 86], [71, 86], [71, 85], [71, 89],
            [72, 89], [72, 90], [71, 90], [75, 82], [75, 83], [76, 83], [76, 82]]

def create_stopper_or():
    # Base stopper required regardless of the OR gate's specific input parameters.
    return [[58, 111], [59, 111], [58, 112], [60, 112], [60, 113], [60, 114], [61, 114]]

def create_stopper_or_01():
    # Intercept block for an OR gate with inputs 0, 1.
    return [[13, 66], [14, 66], [13, 67], [15, 67], [15, 68], [15, 69], [16, 69]]

def create_stopper_or_10():
    # Intercept block for an OR gate with inputs 1, 0.
    return [[13, 106], [14, 106], [13, 107], [15, 107], [15, 108], [15, 109], [16, 109]]

def create_stopper_or_00():
    # Dual intercept block for when neither signal should be generated.
    return [[13, 66], [14, 66], [13, 67], [15, 67], [15, 68], [15, 69], [16, 69], [13, 106], [14, 106], [13, 107], [15, 107], [15, 108], [15, 109], [16, 109]]

# ---------------------------------------------------------
# GAME OF LIFE ENGINE
# ---------------------------------------------------------

def count_living_neighbors(universe_state, row_idx, col_idx):
    # Calculates the amount of surrounding living cells within the immediate 3x3 area.
    total_alive = 0
    for r in range(row_idx - 1, row_idx + 2):
        for c in range(col_idx - 1, col_idx + 2):
            try:
                if universe_state[r, c] == ALIVE_CELL:
                    total_alive += 1
            except IndexError:
                pass
    
    # We must not include the target pixel itself in its own neighbor count.
    if universe_state[row_idx, col_idx] == ALIVE_CELL:
        total_alive -= 1
        
    return total_alive

def compute_next_generation(frame_data):
    # Core function applied continuously to transform the grid into its next logical generation.
    global cellular_universe, animation_matrix
    next_generation_universe = cellular_universe.copy()
    
    for row_idx in range(MAX_ROWS):
        for col_idx in range(MAX_COLS):
            total_alive = count_living_neighbors(cellular_universe, row_idx, col_idx)
            
            # Execute Conway's survival, death, and birth parameters logic.
            if cellular_universe[row_idx, col_idx] == ALIVE_CELL:
                if (total_alive < 2) or (total_alive > 3):
                    next_generation_universe[row_idx, col_idx] = DEAD_CELL
            else:
                if total_alive == 3:
                    next_generation_universe[row_idx, col_idx] = ALIVE_CELL
                    
    animation_matrix.set_data(next_generation_universe)
    cellular_universe = next_generation_universe
    return [animation_matrix]

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == '__main__':
    # Validate user inputs.
    if len(sys.argv) < 3:
        print("Usage: python or_gate.py <input_A> <input_B>")
        sys.exit()

    input_A, input_B = int(sys.argv[1]), int(sys.argv[2])
    coordinates_to_activate = []
    
    # Construct the base OR Gate infrastructure.
    coordinates_to_activate.append(create_forward_glider(1, 0))
    coordinates_to_activate.append(create_forward_glider(1, 40))
    coordinates_to_activate.append(create_forward_glider(1, 80))
    coordinates_to_activate.append(create_backward_glider(3, 120))
    coordinates_to_activate.append(create_stopper_or())
    coordinates_to_activate.append(create_eater_or())
    
    # Spawn conditional stopper mechanisms based on boolean CLI inputs.
    if input_A == 0 and input_B == 0:
        coordinates_to_activate.append(create_stopper_or_00())
    elif input_A == 0 and input_B == 1:
        coordinates_to_activate.append(create_stopper_or_01())
    elif input_A == 1 and input_B == 0:
        coordinates_to_activate.append(create_stopper_or_10())

    # Create the cellular grid and apply initial coordinates.
    cellular_universe = np.zeros((MAX_ROWS, MAX_COLS))
    for pattern_group in coordinates_to_activate:
        for cell_coord in pattern_group:
            cellular_universe[cell_coord[0], cell_coord[1]] = ALIVE_CELL

    # Initiate rendering environment.
    fig, ax = plt.subplots()
    
    # Apply cmap='binary' to force the color palette to white for Dead and black for Alive cells.
    animation_matrix = ax.matshow(cellular_universe, cmap='binary', vmin=DEAD_CELL, vmax=ALIVE_CELL)
    
    ani = animation.FuncAnimation(fig, compute_next_generation, interval=1, save_count=5)
    plt.title(f"OR Gate (A={input_A}, B={input_B})")
    plt.show()