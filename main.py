import math as m
import numpy as np
import matplotlib.pyplot as plt

from plotter import PlotterOT

GM = 6.672e-11 * 5.972e24
R = 6.378e6
W = np.array([0.0, 0.0, 2 * m.pi / (24 * 60 * 60)])

gas_constant = 8.3145
air_molar_mass = 29e-3
air_temperature = 273
air_density_asl = 1.2754


def air_density_at(height):
    du = GM * air_molar_mass * (1 / height - 1 / R)
    return air_density_asl * m.exp(du / gas_constant / air_temperature)


dry_mass = 85e3
fuel_mass = 1200e3
burn_time = 60 * 6
thrust_force = 2.3e6 * 6
mu = fuel_mass / burn_time
area = m.pi * 9 ** 2 / 4
cd = 0.07

dt = 1
T = burn_time
n_steps = int(T / dt)


def simulate_res(a, b):
    global dt, n_steps
    r0 = np.array([0.0, R, 0.0])
    v0 = np.zeros(3)

    mass = dry_mass + fuel_mass

    rs = [r0]
    vs = [v0]

    for step in range(n_steps):
        r = np.array(rs[step])
        v = np.array(vs[step])

        r_abs = np.linalg.norm(r)

        a_grav = -r * GM / r_abs ** 3

        a_thrust = np.zeros(3)

        if mass > dry_mass:
            burn_completion = 1 - (mass - dry_mass) / fuel_mass
            angle = m.pi / 2 + (a * burn_completion) ** b
            thrust_dir = np.array([m.cos(angle), m.sin(angle), 0])
            a_thrust = thrust_dir * thrust_force / mass

            mass -= mu * dt

        a_centrifugal = np.linalg.norm(W) ** 2 * r
        a_coriolis = 2 * np.cross(W, v)

        air_density = air_density_at(r_abs)
        a_drag = cd * -air_density * np.linalg.norm(v) * v * area / 2 / mass

        acceleration = a_grav + a_thrust + a_centrifugal + a_coriolis + a_drag

        v += acceleration * dt
        r += v * dt

        rs.append(r)
        vs.append(v)

    return rs, vs


A_min = 0
A_max = 3
A_n = 50

B_min = 0
B_max = 3
B_n = 50

A = np.linspace(A_min, A_max, A_n)
B = np.linspace(B_min, B_max, B_n)

fig, ax = plt.subplots()

data = np.empty((A_n, B_n))

for i in range(A_n):
    for j in range(B_n):
        state = simulate_res(A[i], B[j])
        launch = PlotterOT(state)
        data[i][j] = launch.get_perigee()
    print(round(i / A_n * 100, 1), '%', sep=' ')
print()

pos_best = np.argmax(data)
a_best = A[pos_best // B_n]
b_best = B[pos_best % B_n]

ax.imshow(data, extent=[B_min, B_max, A_max, A_min])
ax.set_aspect((B_max - B_min) / (A_max - A_min))
ax.set_xlabel(r'$Параметр\ \bf{b}$')
ax.set_ylabel(r'$Параметр\ \bf{a}$')
ax.set_title(r'$a\times b:$ ' + f'{A_n}' + r'$\times$' + f'{B_n}' +
             r'$;\ Шаг:\ $' + f'{dt} ' + r'$с$' + '\n' +
             r'$a_{max}=$' + f'{round(a_best, 2)}' + r'$;\ b_{max}=$' + f'{round(b_best, 2)}')

fig.savefig(f'fields/{A_n}x{B_n}_{A_min}-{A_max}_{B_min}-{B_max}_{dt}s.png', dpi=1000)

best_state = simulate_res(a_best, b_best)
best_launch = PlotterOT(best_state)

print(a_best, b_best)
print(best_launch.get_perigee(), np.linalg.norm(best_launch.ecc))

best_launch.plot_orbit()

best_launch.animate_launch()
