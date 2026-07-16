import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# Define constants for the cell states
ALIVE_CELL = 255
DEAD_CELL = 0
MAX_ROWS = 80
MAX_COLS = 130

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
    # Repurposed eater used in the AND gate to consume output waste.
    return [[72, 82], [73, 82], [73, 83], [73, 84], [73, 85], [72, 86], [71, 86], [71, 85], [71, 89],
            [72, 89], [72, 90], [71, 90], [75, 82], [75, 83], [76, 83], [76, 82]]

def create_stopper_and_01():
    # Signal interceptor block for when A is 0 and B is 1.
    return [[15, 28], [16, 28], [15, 29], [17, 29], [17, 30], [17, 31], [18, 31]]

def create_stopper_and_10():
    # Signal interceptor block for when A is 1 and B is 0.
    return [[14, 68], [15, 68], [14, 69], [16, 69], [16, 70], [16, 71], [17, 71]]

def create_stopper_and_00():
    # Dual signal interceptor block for when both A and B are 0.
    return [[14, 68], [15, 68], [14, 69], [16, 69], [16, 70], [16, 71], [17, 71], [15, 28], [16, 28], [15, 29], [17, 29], [17, 30], [17, 31], [18, 31]]

def create_base_stopper_and():
    # Foundational stopper always required when signals are completely 0.
    return [[56, 55], [56, 56], [57, 56], [58, 55], [58, 54], [58, 53], [59, 53]]

# ---------------------------------------------------------
# GAME OF LIFE ENGINE
# ---------------------------------------------------------

def count_living_neighbors(universe_state, row_idx, col_idx):
    # Determine the number of alive neighbors surrounding a given coordinate.
    total_alive = 0
    for r in range(row_idx - 1, row_idx + 2):
        for c in range(col_idx - 1, col_idx + 2):
            try:
                if universe_state[r, c] == ALIVE_CELL:
                    total_alive += 1
            except IndexError:
                pass
    
    # Exclude the target cell itself from the neighbor count.
    if universe_state[row_idx, col_idx] == ALIVE_CELL:
        total_alive -= 1
        
    return total_alive

def compute_next_generation(frame_data):
    # Calculates the grid state for the next visual frame.
    global cellular_universe, animation_matrix
    next_generation_universe = cellular_universe.copy()
    
    for row_idx in range(MAX_ROWS):
        for col_idx in range(MAX_COLS):
            total_alive = count_living_neighbors(cellular_universe, row_idx, col_idx)
            
            # Apply Conway's conditions for life and death.
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
    # Validate user inputs (requires two binary arguments).
    if len(sys.argv) < 3:
        print("Usage: python and_gate.py <input_A> <input_B>")
        sys.exit()

    input_A, input_B = int(sys.argv[1]), int(sys.argv[2])
    coordinates_to_activate = []
    
    # Establish the default mechanical parts of the AND Gate.
    coordinates_to_activate.append(create_forward_glider(1, 0))
    coordinates_to_activate.append(create_forward_glider(0, 40))
    coordinates_to_activate.append(create_backward_glider(3, 90))
    coordinates_to_activate.append(create_eater_or())

    # Branching logic: we deploy specific blockades based on the boolean inputs.
    if input_A == 0 and input_B == 0:
        coordinates_to_activate.append(create_stopper_and_00())
        coordinates_to_activate.append(create_base_stopper_and())
    elif input_A == 0 and input_B == 1:
        coordinates_to_activate.append(create_stopper_and_01())
    elif input_A == 1 and input_B == 0:
        coordinates_to_activate.append(create_stopper_and_10())

    # Initialize and populate the grid based on aggregated patterns.
    cellular_universe = np.zeros((MAX_ROWS, MAX_COLS))
    for pattern_group in coordinates_to_activate:
        for cell_coord in pattern_group:
            cellular_universe[cell_coord[0], cell_coord[1]] = ALIVE_CELL

    # Prepare graphics.
    fig, ax = plt.subplots()
    
    # Using 'binary' colormap sets 0 -> white and 255 -> black.
    animation_matrix = ax.matshow(cellular_universe, cmap='binary', vmin=DEAD_CELL, vmax=ALIVE_CELL)
    
    ani = animation.FuncAnimation(fig, compute_next_generation, interval=1, save_count=5)
    plt.title(f"AND Gate (A={input_A}, B={input_B})")
    plt.show()

    import os

# ---------------------------------------------------------
# PATTERN GENERATORS
# Hämtade från våra tidigare spawner-funktioner
# ---------------------------------------------------------

def create_forward_glider(row_offset, col_offset):
    glider_coords = [[5, 1],[6, 1],[6, 2],[5, 2],[5, 5],[4, 6],[5, 6],[6, 6],[3, 7],[7, 7],[5, 8],[2, 9],[2, 10],
                     [3, 11], [4, 12],[5, 12],[6, 12],[7, 11],[8, 10],[8, 9],[7, 17],[7, 20],[7, 21],[7, 22],[6, 22],
                     [5, 21],[6, 26],[5, 26],[1, 26],[0, 26],[1, 28],[5, 28],[2, 29],[3, 29],[4, 29],[4, 30],[3, 30],
                     [2, 30],[3, 35],[4, 35],[4, 36],[3, 36]]
    return [[r + row_offset, c + col_offset] for r, c in glider_coords]

def create_backward_glider(row_offset, col_offset):
    reverse_glider_coords = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 6], [1, 6], [1, 7], [0, 7], [-1, 7], [-1, 6], [-2, 8], 
                             [-2, 10], [-3, 10], [2, 8], [2, 10], [3, 10], [3, 14], [4, 14], [4, 15], [4, 16], [2, 15], 
                             [4, 19], [4, 25], [3, 24], [2, 24], [1, 24], [0, 25], [-1, 26], [-1, 27], [5, 26], [5, 27], 
                             [2, 28], [2, 30], [2, 31], [1, 30], [3, 30], [0, 29], [4, 29], [2, 34], [2, 35], [3, 35], [3, 34]]
    return [[r + row_offset, c + col_offset] for r, c in reverse_glider_coords]

def create_eater_not():
    return [[32, 24], [32, 25], [33, 24], [33, 24], [33, 25], [32, 28], [32, 29], [33, 28],
            [34, 29], [34, 30], [34, 31], [34, 32], [33, 32], [36, 31], [36, 32], [37, 32], [37, 31]]

def create_eater_or():
    return [[72, 82], [73, 82], [73, 83], [73, 84], [73, 85], [72, 86], [71, 86], [71, 85], [71, 89],
            [72, 89], [72, 90], [71, 90], [75, 82], [75, 83], [76, 83], [76, 82]]

def create_stopper_or():
    return [[58, 111], [59, 111], [58, 112], [60, 112], [60, 113], [60, 114], [61, 114]]

# ---------------------------------------------------------
# FILSKRIVAR-LOGIK
# ---------------------------------------------------------

def export_gate_to_txt(filename, component_list):
    """
    Samlar alla koordinater från komponenterna, tar bort dubbletter, 
    sorterar dem och skriver dem till en fil i formatet (x,y).
    """
    all_coordinates = set()
    for component in component_list:
        for r, c in component:
            all_coordinates.add((r, c))
            
    with open(filename, 'w') as file:
        for r, c in sorted(all_coordinates):
            # Formatet som krävs av labben: (0,0)
            file.write(f"({r},{c})\n")
    print(f"Skapade {filename} med {len(all_coordinates)} levande celler.")

if __name__ == '__main__':
    print("Genererar textfiler med startkoordinater enligt labbinstruktionerna...")

    # 1. NOT-grinden (Startuppställning)
    not_components = [
        create_forward_glider(1, 0),
        create_backward_glider(3, 40),
        create_eater_not()
    ]
    export_gate_to_txt("not_gate.txt", not_components)

    # 2. AND-grinden (Startuppställning)
    and_components = [
        create_forward_glider(1, 0),
        create_forward_glider(0, 40),
        create_backward_glider(3, 90),
        create_eater_or() # Används för att äta output
    ]
    export_gate_to_txt("and_gate.txt", and_components)

    # 3. OR-grinden (Startuppställning)
    or_components = [
        create_forward_glider(1, 0),
        create_forward_glider(1, 40),
        create_forward_glider(1, 80),
        create_backward_glider(3, 120),
        create_stopper_or(),
        create_eater_or()
    ]
    export_gate_to_txt("or_gate.txt", or_components)

    print("Klar! Filerna ligger redo att lämnas in.")