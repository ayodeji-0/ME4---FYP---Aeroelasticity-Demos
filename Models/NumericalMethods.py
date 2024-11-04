#   Numerical methods for solving ODEs and integrals

import numpy as np


# Simpson's Rule - modified trapazoidal rule
def simpsons_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + 2 * np.sum(y[2:2*n:2]) + 4 * np.sum(y[1:2*n:2]) + y[2*n])

# Gauss-Legendre Quadrature
def gauss_legendre_quadrature(f, a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * np.sum(w * f(0.5 * (b - a) * x + 0.5 * (b + a)))

# Runge-Kutta 4th Order Method
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Runge-Kutta 2nd Order Method
def rk2(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h, y + k1)
    return y + 0.5 * (k1 + k2)

