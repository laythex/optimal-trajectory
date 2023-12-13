import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from solver import GM, R, cross


def rv2pecc(r, v):
    k = cross(r, v)
    p = np.linalg.norm(k) ** 2 / GM
    ecc = -cross(k, v) / GM - r / np.linalg.norm(r)
    return p, ecc


def get_ellipse(p, ecc):
    theta = np.linspace(0, 2 * np.pi, 3600)
    theta0 = m.atan2(ecc[1], ecc[0])
    rho = p / (1 + np.linalg.norm(ecc) * np.cos(theta - theta0))
    return pol2cart(rho, theta)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


class Launch:
    def __init__(self, state):
        self.rs = np.transpose(state[:3])
        self.vs = np.transpose(state[3:])
        self.count = state.shape[0]

        self.rf = self.rs[-1]
        self.vf = self.vs[-1]

        self.infinite = np.linalg.norm(self.vf) ** 2 / 2 - GM / np.linalg.norm(self.rf) >= 0

        self.p, self.ecc = rv2pecc(self.rf, self.vf)

        self.fig, self.ax = None, None
        self.line_r = None
        self.line_v = None
        self.line_o = None
        self.line_t = None
        self.anim = None

    def get_final_rv(self):
        return self.rf, self.vf

    def get_perigee(self):
        if self.infinite:
            return -R * 1e-3

        per = self.p / (1 + np.linalg.norm(self.ecc))
        return (per - R) * 1e-3

    def plot_orbit(self):
        if self.infinite:
            return

        self.fig, self.ax = plt.subplots()

        self.ax.scatter(*self.rf[0:2], color='black')
        self.ax.scatter(*(self.rf + self.vf * 1e2)[0:2], color='black')

        self.ax.plot(*get_ellipse(*rv2pecc(self.rf, self.vf)), color='black')
        self.ax.plot(*get_ellipse(R, np.zeros(3)), color='blue')

        self.ax.set_xlim(-2 * R, 2 * R)
        self.ax.set_ylim(-2 * R, 2 * R)

        self.ax.set_aspect('equal')

        self.ax.set_xlabel(r'$x,\ km$')
        self.ax.set_ylabel(r'$y,\ km$')

        self.ax.set_title(r'$Pe=$' + f'{round(self.get_perigee(), 1)} ' + r'$km;\ $' +
                          r'$Ecc=$' + f'{round(np.linalg.norm(self.ecc), 4)}')

    def animate_launch(self, save=False):
        self.line_r, = self.ax.plot([], [], 'ro')
        self.line_v, = self.ax.plot([], [], 'ro')
        self.line_o, = self.ax.plot([], [], 'r--')
        self.line_t, = self.ax.plot([], [], 'r')

        self.anim = FuncAnimation(self.fig, self.update, self.count, interval=0.1)

        if save:
            self.anim.save('launch.gif')

        plt.show()

    def update(self, k):
        self.line_r.set_data(*self.rs[k][0:2])
        self.line_v.set_data(*(self.rs[k] + self.vs[k] * 1e2)[0:2])
        self.line_o.set_data(*get_ellipse(*rv2pecc(self.rs[k], self.vs[k])))
        self.line_t.set_data(*(np.split(self.rs[:k], 3, 1))[0:2])

        return self.line_r, self.line_v, self.line_o, self.line_t,
