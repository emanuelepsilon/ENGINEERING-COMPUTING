import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import matplotlib.colors as mcolors

#Depending on how fast you want the animation to run you can change the interval value = t
t = 100

class GHCA_Simulator:
    def __init__(self, n=50, N_states=5, e_param=2, initial_file=None):
        self.n = n
        self.N = N_states  # Total states 0 to N-1
        self.e = e_param   # Excited if 1 <= state <= e
        self.steps = 0
        self.running = True
        
        # Cycle Detection
        self.history = {}  # {hash: step_index}
        self.transient_time = None
        self.period = None
        
        # Initialization
        if initial_file:
            self.grid = self.load_from_file(initial_file)
        else:
            self.grid = np.random.randint(0, self.N, (n, n))
            
        self.setup_plot()

    def load_from_file(self, filename):
        """Placeholder for reading configuration from a file."""
        try:
            return np.loadtxt(filename, dtype=int)
        except Exception as e:
            print(f"Error loading file: {e}. Falling back to random.")
            return np.random.randint(0, self.N, (self.n, self.n))

    def save_to_file(self, filename="final_config.txt"):
        """Saves the current grid to a text file."""
        np.savetxt(filename, self.grid, fmt='%d')
        print(f"Configuration saved to {filename}")

    def get_next_state(self):
        """Vectorized update"""
        new_grid = np.zeros_like(self.grid)
        
        # Identify Excited Bit/boolian-Mask (1 <= state <= e)
        is_excited = (self.grid >= 1) & (self.grid <= self.e)
        
        # Von Neumann Neighborhood check 
        # Shifted grids: neighbors outside are 0 (not excited) due to padding
        up    = np.pad(is_excited, ((0,1),(0,0)), mode='wrap')[1:, :]
        down  = np.pad(is_excited, ((1,0),(0,0)), mode='wrap')[:-1, :]
        left  = np.pad(is_excited, ((0,0),(0,1)), mode='wrap')[:, 1:]
        right = np.pad(is_excited, ((0,0),(1,0)), mode='wrap')[:, :-1]
        
        neighbor_excited = up | down | left | right
        
        # Rule 1: State 0 transitions
        mask_0 = (self.grid == 0)
        new_grid[mask_0 & neighbor_excited] = 1
        new_grid[mask_0 & ~neighbor_excited] = 0
        
        # Rule 2: 1 <= s <= N-2 transitions
        mask_advance = (self.grid >= 1) & (self.grid <= self.N - 2)
        new_grid[mask_advance] = self.grid[mask_advance] + 1
        
        # Rule 3: s = N-1 transitions to 0
        mask_reset = (self.grid == self.N - 1)
        new_grid[mask_reset] = 0
        
        return new_grid

    def check_cycle(self):
        """Detects transient time and period using state hashing."""
        if self.period is not None:
            return

        state_hash = hash(self.grid.tobytes())
        if state_hash in self.history:
            self.transient_time = self.history[state_hash]
            self.period = self.steps - self.transient_time
            print(f"Cycle Detected! Transient: {self.transient_time}, Period: {self.period}")
        else:
            self.history[state_hash] = self.steps

    def update(self, frame):
        if self.running:
            self.check_cycle()
            self.grid = self.get_next_state()
            self.steps += 1
            self.im.set_array(self.grid)
            
            # Update Overlay
            status = f"Step: {self.steps}"
            if self.period:
                status += f" | Transient: {self.transient_time} | Period: {self.period}"
            self.ax.set_title(status)
            
        return [self.im]

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Custom discrete colormap
        cmap = plt.get_cmap('viridis', self.N)
        self.im = self.ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=self.N-1)
        
        # UI Buttons
        ax_pause = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.btn_pause = Button(ax_pause, 'Start/Pause')
        self.btn_pause.on_clicked(self.toggle_pause)

    def toggle_pause(self, event):
        self.running = not self.running
    
    def run(self):
        ani = FuncAnimation(self.fig, self.update, interval= t, save_count=100)
        plt.show()
        self.save_to_file()


if __name__ == "__main__":
    # Test with total off N=5 states, where 1-2 are excited and 3-4 are recovery
    # Swap initial_file = None to "filename" if adding an initial file
    sim = GHCA_Simulator(n=50, N_states=5, e_param=2, initial_file= None)
    sim.run()
    