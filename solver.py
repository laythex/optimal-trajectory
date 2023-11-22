import math as m
import numpy as np

# Astronomical constants
GM = 6.672e-11 * 5.972e24
R = 6.378e6
W = np.array([0.0, 0.0, 2 * m.pi / (24 * 60 * 60)])

# Atmospheric constants
gas_constant = 8.3145
air_molar_mass = 29e-3
air_temperature = 273
air_density_asl = 1.2754


def get_air_density_at(height):
    du = GM * air_molar_mass * (1 / height - 1 / R)
    return air_density_asl * m.exp(du / gas_constant / air_temperature)


# Rocket parameters
dry_mass = 85e3
fuel_mass = 1200e3
burn_time = 60 * 6
thrust_force = 2.3e6 * 6
mu = fuel_mass / burn_time
area = m.pi * 9 ** 2 / 4
cd = 0.07


def calculate_solution(a, b, dt):
    n_steps = int(burn_time / dt)

    r0 = np.array([0.0, R, 0.0])
    v0 = np.zeros(3)

    mass = dry_mass + fuel_mass

    rs = np.zeros((n_steps + 1, 3))
    vs = np.zeros((n_steps + 1, 3))
    rs[0], vs[0] = r0, v0

    for step in range(n_steps):
        r = rs[step]
        v = vs[step]

        r_abs = np.linalg.norm(r)

        a_grav = -r * GM / r_abs ** 3

        a_thrust = np.zeros(3)

        if mass > dry_mass:
            burn_completion = 1 - (mass - dry_mass) / fuel_mass
            angle = m.pi / 2 + (a * burn_completion) ** b
            thrust_dir = np.array([m.cos(angle), m.sin(angle), 0])
            a_thrust = thrust_dir * thrust_force / mass

            mass -= mu * dt

        a_centrifugal = np.cross(W, np.cross(W, r))
        a_coriolis = 2 * np.cross(W, v)

        air_density = get_air_density_at(r_abs)
        a_drag = -cd * air_density * np.linalg.norm(v) * v * area / 2 / mass

        acceleration = a_grav + a_thrust + a_centrifugal + a_coriolis + a_drag

        v += acceleration * dt
        r += v * dt

        rs[step + 1] = r
        vs[step + 1] = v

    return rs, vs
