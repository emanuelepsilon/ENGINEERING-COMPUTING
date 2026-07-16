import roadster
import matplotlib.pyplot as plt
import numpy as np

def ordo(n,p):
   return 1/(n**p)


route = 'speed_anna'
distances, _ = roadster.load_route(route)
distance = distances[-1]
values = []
n = []
for i in np.arange(0, 30, 1):
   c = roadster.time_to_destination(distance, route, 2 ** i)
   values.append(c)
   n.append(2 ** i)
values = np.array(values)
# Vi kan nu beräkna felet och den "riktiga" integralen
thirdEstimation = values[-1] + ( ( values[-2] - values[-1] ) / 3)
errors = np.array(abs(values - thirdEstimation))
plt.plot(n, errors, color = 'red', label = 'Uppskattat fel')
help_line = ordo(np.array(n), 2)
help_line = np.dot(help_line, errors[0])
plt.plot(n, help_line, color = 'green', label = '1/n^2')
plt.title('Felapproximation för Anna')
plt.legend()
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.show()