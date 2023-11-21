import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

GM = 6.672e-11 * 5.972e24
R = 6.371e6


def mag(vector):
    return m.sqrt(np.dot(vector, vector))


def rv2pecc(r, v):
    k = np.cross(r, v)
    p = mag(k) ** 2 / GM
    ecc = -np.cross(k, v) / GM - r / mag(r)
    return p, ecc


def get_ellipse(p, ecc):
    theta = np.linspace(0, 2 * np.pi, 3600)
    theta0 = m.atan2(ecc[1], ecc[0])
    rho = p / (1 + mag(ecc) * np.cos(theta - theta0))
    return pol2cart(rho, theta)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


# noinspection PyAttributeOutsideInit
class PlotterOT:
    def __init__(self, _states):
        self.rs = np.array(_states[0])
        self.vs = np.array(_states[1])

        self.rf = self.rs[-1]
        self.vf = self.vs[-1]

        self.infinite = mag(self.vf) ** 2 / 2 - GM / mag(self.rf) >= 0

        self.p, self.ecc = rv2pecc(self.rf, self.vf)

    def get_final_rv(self):
        return self.rf, self.vf

    def get_perigee(self):
        if self.infinite:
            return -R * 1e-3

        per = self.p / (1 + mag(self.ecc))
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
                          r'$Ecc=$' + f'{round(mag(self.ecc), 4)}')

    def animate_launch(self):
        self.line_r, = self.ax.plot([], [], 'ro')
        self.line_v, = self.ax.plot([], [], 'ro')
        self.line_o, = self.ax.plot([], [], 'r--')
        self.line_t, = self.ax.plot([], [], 'r')
        self.n = len(self.rs)

        self.anim = FuncAnimation(self.fig, self.update, self.n, interval=0.1)
        # self.anim.save('launch2.gif')
        plt.show()

    def update(self, k):
        self.line_r.set_data(*self.rs[k][0:2])
        self.line_v.set_data(*(self.rs[k] + self.vs[k] * 1e2)[0:2])
        self.line_o.set_data(*get_ellipse(*rv2pecc(self.rs[k], self.vs[k])))
        self.line_t.set_data(*(np.split(self.rs[:k], 3, 1))[0:2])

        return self.line_r, self.line_v, self.line_o, self.line_t,
