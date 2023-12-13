import math as m
import numpy as np
from scipy.integrate import solve_ivp

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


# IDE bug workaround
def cross(a, b):
    return np.cross(a, b)


def ivp_rhs(_, state, a, b):
    r = state[:3]
    v = state[3:6]
    mass = state[-1]

    r_abs = np.linalg.norm(r)

    acc_grav = -r * GM / r_abs ** 3

    burn_completion = 1 - (mass - dry_mass) / fuel_mass
    angle = 0.5 * m.pi + (a * burn_completion) ** b
    thrust_dir = np.array([m.cos(angle), m.sin(angle), 0])
    acc_thrust = thrust_dir * thrust_force / mass

    acc_centrifugal = -cross(W, cross(W, r))
    acc_coriolis = -2 * cross(W, v)

    air_density = get_air_density_at(r_abs)
    acc_drag = -0.5 * cd * air_density * np.linalg.norm(v) * v * area / mass

    acc = acc_grav + acc_thrust + acc_centrifugal + acc_coriolis + acc_drag

    return np.concatenate((v, acc, np.array([-mu])))


def calculate_solution(a, b, max_step):
    r0 = np.array([0.0, R, 0.0])
    v0 = np.zeros(3)
    mass0 = dry_mass + fuel_mass
    state0 = np.concatenate((r0, v0, np.array([mass0])))

    solution = solve_ivp(ivp_rhs, [0, fuel_mass / mu], state0, max_step=max_step, args=(a, b))

    return solution.y[:6]
