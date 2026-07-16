import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def load_route(route):
    """ 
    Get speed data from route .npz-file. Example usage:

      distance_km, speed_kmph = load_route('speed_anna.npz')
    
    The route file should contain two arrays, distance_km and 
    speed_kmph, of equal length with position (in km) and speed 
    (in km/h) along route. Those two arrays are returned by this 
    convenience function.
    """
    # Read data from npz file
    if not route.endswith('.npz'):
        route = f'{route}.npz' 
    data = np.load(route)
    distance_km = data['distance_km']
    speed_kmph = data['speed_kmph']    
    return distance_km, speed_kmph

def save_route(route, distance_km, speed_kmph):
    """ 
    Write speed data to route file. Example usage:

      save_route('speed_olof.npz', distance_km, speed_kmph)
    
    Parameters have same meaning as for load_route
    """ 
    np.savez(route, distance_km=distance_km, speed_kmph=speed_kmph)

### PART 1A ###
def consumption(v):
    coeff = np.array([546.8, 50.31, 0.2584, 0.008210]) # Givna koefficienter för Tesla Roadster
    return coeff[0]*v**-1 + coeff[1] + coeff[2]*v + coeff[3]*v**2

### PART 1B ###
def velocity(x, route):
    # ALREADY IMPLEMENTED!
    """
    Interpolates data in given route file, and evaluates the function
    in x
    """
    # Load data
    distance_km, speed_kmph = load_route(route)
    # Check input ok?
    assert np.all(x>=0), 'x must be non-negative'
    assert np.all(x<=distance_km[-1]), 'x must be smaller than route length'
    # Interpolate
    v = interpolate.pchip_interpolate(distance_km, speed_kmph,x)
    return v

### PART 2A ###
def time_to_destination(x, route, n):
    # integrera längs velocity från 0 till x, med n steg
    h = x/n # steglängd, givet att vi alltid börjar vid 0
    v = velocity(np.linspace(0, x, n + 1), route) # Skapa en interpolerad hastighet, med punkter vid varje kant av våra trapezoider
    integrand = 1 / v
    return h * (np.sum(integrand) - (integrand[0] + integrand[-1]) / 2)

### PART 2B ###
def total_consumption(x, route, n):
    h = x/n # steglängd, givet att vi börjar vid 0
    v = velocity(np.linspace(0, x, n + 1), route) # interpolerad hastighet, till x med steglängd h
    c = consumption(v)
    return h * (np.sum(c) - (c[0] + c[-1]) / 2)

### PART 3A ###
def distance(T, route): 
    # REMOVE THE FOLLOWING LINE AND WRITE YOUR SOLUTION
    raise NotImplementedError('distance not implemented yet!')

### PART 3B ###
def reach(C, route):
    # REMOVE THE FOLLOWING LINE AND WRITE YOUR SOLUTION
    raise NotImplementedError('reach not implemented yet!')
