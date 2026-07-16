import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# Tillståndskonstanter
READY = 0
FIRING = 1
RESTING = 2

def count_firing_neighbors(grid):
    """Räknar antalet "firing"-grannar med periodiska randvillkor (Moore-grannskap)."""
    firing = (grid == FIRING).astype(int)
    neighbors = (
        np.roll(firing, 1, axis=0) + np.roll(firing, -1, axis=0) +
        np.roll(firing, 1, axis=1) + np.roll(firing, -1, axis=1) +
        np.roll(np.roll(firing, 1, axis=0), 1, axis=1) +
        np.roll(np.roll(firing, -1, axis=0), 1, axis=1) +
        np.roll(np.roll(firing, 1, axis=0), -1, axis=1) +
        np.roll(np.roll(firing, -1, axis=0), -1, axis=1)
    )
    return neighbors

def update(frameNum, img, grid):
    new_grid = grid.copy()
    firing_neighbors = count_firing_neighbors(grid)

    # De tre reglerna från lab-PM:et
    new_grid[(grid == READY) & (firing_neighbors == 2)] = FIRING
    new_grid[grid == FIRING] = RESTING
    new_grid[grid == RESTING] = READY

    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img,

def run_simulation(grid, N, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Svart = Ready, Gul = Firing, Blå = Resting
    cmap = ListedColormap(['black', 'yellow', 'royalblue'])
    
    img = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')

    ani = animation.FuncAnimation(
        fig, update, fargs=(img, grid),
        frames=200, interval=150, blit=True
    )
    plt.show()

# --- EXAKTA STARTVILLKOR FÖR ATT UPPFYLLA KRAVEN ---

def task_a_moving_pattern(N):
    """
    Krav: 'pattern moving forward in time'.
    Lösning: En 2-celler bred "Glider". Eftersom den är exakt 2 celler bred, 
    kommer cellerna rakt framför att se exakt 2 aktiva grannar och tändas.
    Den kommer att röra sig i en rak linje över skärmen i all oändlighet.
    """
    grid = np.full((N, N), READY)
    c = N // 2
    grid[c, c:c+2] = FIRING   # Fronten
    grid[c+1, c:c+2] = RESTING # Svansen (tvingar rörelsen uppåt)
    return grid, "Task A: Stable Moving Glider"

def task_b_moving_and_creating(N):
    """
    Krav: 'moves forward and creates new moving patterns along the way'.
    Lösning: "The Carrier". En stor V-formad rymdfarkost. Fronten är en 
    perfekt glidare som drar skeppet uppåt. Inuti V-skölden ligger en 
    instabil last som ständigt interagerar med vingarna och kastar ur sig 
    nya, mindre glidare bakåt i sitt kölvatten.
    """
    grid = np.full((N, N), READY)
    c = N // 2
    
    # 1. Fronten (En stabil glidare som drar allt uppåt)
    grid[c, c] = FIRING
    grid[c, c+1] = FIRING
    grid[c+1, c] = RESTING
    grid[c+1, c+1] = RESTING
    
    # 2. Sköldarna/Vingarna (Diagonala linjer som skyddar lasten)
    for i in range(1, 10):
        # Höger vinge
        grid[c+i, c+1+i] = FIRING
        grid[c+i+1, c+1+i] = RESTING
        # Vänster vinge
        grid[c+i, c-i] = FIRING
        grid[c+i+1, c-i] = RESTING

    # 3. Lasten (En instabil reaktor inuti skeppet)
    grid[c+4, c] = FIRING
    grid[c+4, c+1] = FIRING
    grid[c+5, c] = RESTING
    grid[c+6, c+1] = FIRING
    
    return grid, "Task B: The Puffer Carrier"

def task_c_periodic_motion(N):
    """
    Krav: 'periodic motion'.
    Lösning: En lodrät "glidare" som rör sig åt höger. När den når höger kant 
    åker den ut och kommer in på vänster sida tack vare de periodiska randvillkoren. 
    Hela systemet har därmed en period på N steg.
    """
    grid = np.full((N, N), READY)
    
    # Vi placerar mönstret i mitten för att tydligt se hur det rör sig
    # mot kanten, försvinner ut till höger, och dyker upp till vänster.
    c = N // 2
    
    # En glidare som är 2 celler hög
    # FIRING ligger till höger (i rörelseriktningen)
    grid[c:c+2, c] = FIRING      
    
    # RESTING ligger direkt till vänster om FIRING (knuffar den framåt)
    grid[c:c+2, c-1] = RESTING   
    
    return grid, "Task C: Periodic Motion (Boundary Wrap-around)"

if __name__ == '__main__':
    N = 40 # Rutnätets storlek
    
    # =========================================================
    # VÄLJ UPPGIFT HÄR! Ändra TASK till 'A', 'B' eller 'C'
    TASK = "A"
    # =========================================================
    
    if TASK == 'A':
        grid, title = task_a_moving_pattern(N)
    elif TASK == 'B':
        grid, title = task_b_moving_and_creating(N)
    elif TASK == 'C':
        grid, title = task_c_periodic_motion(N)
        
    run_simulation(grid, N, title)