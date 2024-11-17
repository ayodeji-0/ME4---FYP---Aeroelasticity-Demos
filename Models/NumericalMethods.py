#   Numerical methods for solving ODEs and integrals

import numpy as np

## Integration Methods
# From aeroelasticity textbook, maths tools section for integrals

# Trapezoidal Rule
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 2 * (y[0] + 2 * np.sum(y[1:n]) + y[n])

# Simpson's Rule - modified trapazoidal rule
def simpsons_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + 2 * np.sum(y[2:2*n:2]) + 4 * np.sum(y[1:2*n:2]) + y[2*n])

# Need to correct the following functions
# Multhopps Quadrature - integral from b to a of f(x) dx =  (b-a/m+1) sum f(phi_i) sin(phi_i) for i = 1 to m
def multhopps_quadrature(f, a, b, m):
    phi = np.pi * np.arange(1, m + 1) / (m + 1)
    return (b - a) / (m + 1) * np.sum(f((b - a) / 2 * np.sin(phi) + (b + a) / 2) * np.sin(phi))

# Lagrange Interpolation
def lagrange_interpolation(x, y, t):
    n = len(x)
    L = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i] *= (t - x[j]) / (x[i] - x[j])
    return np.sum(y * L)


# Not from the textbook

# Gauss-Legendre Quadrature
def gauss_legendre_quadrature(f, a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * np.sum(w * f(0.5 * (b - a) * x + 0.5 * (b + a)))


## Interpolation Methods

# Newton's Divided Difference
def newtons_divided_difference(x, y):
    n = len(x)
    a = y.copy()
    for i in range(1, n):
        a[i:n] = (a[i:n] - a[i - 1]) / (x[i:n] - x[i - 1])
    return a



## ODE Solvers

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

## Combined Integration Class

# Singular class for numerical integration by weighting numbers in a given interval for a given array of function values
class weighted_integrals:
    """
    Class for numerical integration by weighting numbers in a given interval for a given array of function values and a given time step

    Attributes
    ----------
    method : str
        Method of integration
        Defaults to simpsons rule if not specified; computationally efficient for small number of intervals
    y : array
        Array of y values to integrate over

        or 

    f : function
        Function to be integrated 


    a : float1 
        Lower bound of the interval
        Defaults to 0 if not specified
    b : float2
        Upper bound of the interval
        No default value
    lambda : int
        Number of intervals/timesteps
        Defaults to 100 if not specified or calculated from the length of f
    h : float
        Time step
        Defaults to (b - a) / lambda if not specified
    
    Methods
    -------
    integrate()
        Integrates the function over the interval 
    
    """

    def __init__(self, method = 'simpsons', y = None, f = None, a = 0, b = None, lambda_ = None, h = None):
        self.method = method
        self.y = y
        self.f = f
        self.a = a
        self.b = b
        self.lambda_ = lambda_
        self.h = h

        if self.method == 'simpsons':
            if self.lambda_ == None:
                self.lambda_ = 100
            if self.h == None:
                self.h = (self.b - self.a) / self.lambda_

        if self.method == 'gauss-legendre':
            if self.lambda_ == None:
                self.lambda_ = 5

    def integrate(self):
        if self.method == 'simpsons':
            return simpsons_rule(self.f, self.a, self.b, self.lambda_)
        if self.method == 'gauss-legendre':
            return gauss_legendre_quadrature(self.f, self.a, self.b, self.lambda_)

    def integrate_y(self):
        if self.method == 'simpsons':
            return simpsons_rule(self.y, self.a, self.b, self.lambda_)
        if self.method == 'gauss-legendre':
            return gauss_legendre_quadrature(self.y, self.a, self.b, self.lambda_)