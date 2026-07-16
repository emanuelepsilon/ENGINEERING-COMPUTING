import matplotlib.pyplot as plt
import roadster
import numpy as np


time = 0
steps = 100
route = 'speed_elsa'
estimated_integral = 10
distance_km, speed_kmh = roadster.load_route(route)
length = distance_km[-1]
x = np.linspace(0, distance_km[-1], 100000)
print(x)
'''plt.scatter(x, roadster.velocity(x, route), s = 1, label = 'Interpolerad data')
plt.scatter(distance_km, speed_kmh, s = 30, label = 'Insamlad data')
plt.legend()
plt.grid()
plt.show()'''
while int(estimated_integral) != 57:
    steps += 2
    time = roadster.time_to_destination(length, route, steps) * 60 
    T2h = roadster.time_to_destination(length, route, int((steps / 2))) * 60
    error = ( T2h - time ) / 3
    estimated_integral = time - error
print(f'calculated: {time}, T2h = {T2h}, error: {error}, estimated integral: {estimated_integral}, it took {steps} steps')