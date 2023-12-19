import numpy as np
import matplotlib.pyplot as plt

from solver import calculate_solution
from plotter import Launch

max_step = 1

a_min = 0
a_max = 3
a_count = 50

b_min = 0
b_max = 3
b_count = 50

A = np.linspace(a_min, a_max, a_count)
B = np.linspace(b_min, b_max, b_count)

fig, ax = plt.subplots()

data = np.empty((a_count, b_count))

for i in range(a_count):
    for j in range(b_count):
        launch = Launch(calculate_solution(A[i], B[j], max_step))
        data[i][j] = launch.get_perigee()
    print(f'{round(i / a_count * 100, 1)} %')

a_optimal_index, b_optimal_index = np.unravel_index(np.argmax(data), data.shape)
a_optimal, b_optimal = A[a_optimal_index], B[b_optimal_index]

ax.imshow(data, extent=[b_min, b_max, a_max, a_min])
ax.set_aspect((b_max - b_min) / (a_max - a_min))
ax.set_xlabel(r'$Параметр\ \bf{b}$')
ax.set_ylabel(r'$Параметр\ \bf{a}$')
ax.set_title(r'$a\times b:$ ' + f'{a_count}' + r'$\times$' + f'{b_count}' +
             r'$;\ Шаг:\ $' + f'{max_step} ' + r'$с$' + '\n' +
             r'$a_{0}=$' + f'{round(float(a_optimal), 2)}' + r'$;\ b_{0}=$' + f'{round(float(b_optimal), 2)}')

fig.savefig(f'results/{a_count}x{b_count}_{a_min}-{a_max}_{b_min}-{b_max}_{max_step}s.png', dpi=1000)

optimal_state = Launch(calculate_solution(a_optimal, b_optimal, max_step))
optimal_state.plot_orbit()
optimal_state.animate_launch()
