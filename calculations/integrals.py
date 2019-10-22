# %%

import sympy as sp
# sp.init_printing()

dy = sp.symbols('dy', real=True)

# %% 

exponential = sp.exp(sp.I * 2 * dy ** 2)

analytic_result = float(sp.Abs(sp.integrate(exponential, (dy, -sp.oo, sp.oo)).n()))

#%%

import numpy as np
lims = np.arange(0, 2, 0.01)
integrals = []

for lim in lims:
    c_value = sp.integrate(exponential, (dy, -lim, lim))
    r_value = sp.Abs(c_value.n())
    integrals.append(float(r_value))

#%%

import matplotlib.pyplot as plt
plt.style.use('seaborn')

plt.plot(lims, integrals, label = "Integral")
plt.axhline(analytic_result, c="g", label="sqrt(pi/2)")
plt.xlabel("symmetric integration limits [adimensional: deltaX / (hbar * T / m)]")
plt.legend()
plt.savefig("integral.pdf", format="pdf")
# %%

from scipy.interpolate import interp1d
from scipy.optimize import newton
from scipy.special import erf

spline = interp1d(lims, integrals, kind = "cubic")

centeredspline = lambda x: spline(x) - erf(1/np.sqrt(2))*np.sqrt(np.pi / 2)

root = newton(centeredspline, x0=1./2.)

# %%

root

#%%

# full_integral = sp.Abs(sp.integrate(exponential, (dy, -sp.oo, sp.oo)))

# x = sp.symbols('x')
# value = 0.69

# x_integral = sp.integrate(exponential, (dy, -x, x))

# ident = sp.Eq(x_integral*sp.conjugate(x_integral), (full_integral * value)**2)


#%%
# sp.solve(ident, x)

#%%
