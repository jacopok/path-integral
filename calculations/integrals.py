# %%

import sympy as sp
# sp.init_printing()

dy = sp.symbols('dy', real=True)

# %% 

exponential = sp.exp(sp.I * dy ** 2)
real_exponential = sp.exp( -dy ** 2)

analytic_result = float(sp.Abs(sp.integrate(exponential, (dy, -sp.oo, sp.oo)).n()))

#%%

import numpy as np
lims = np.arange(0, 4, 0.01)
integrals = []
real_integrals = []

for lim in lims:
    c_value = sp.integrate(exponential, (dy, -lim, lim))
    r_value = sp.Abs(c_value.n())
    real_int = sp.integrate(real_exponential, (dy, -lim, lim))
    integrals.append(float(r_value))
    real_integrals.append(real_int)

#%%

import matplotlib.pyplot as plt
plt.style.use('seaborn')
from scipy.special import erf

plt.plot(lims, integrals, label = "Imaginary integral", )
plt.plot(lims, real_integrals, label = "Real integral")
plt.axhline(analytic_result, label="result at infinity", c="indigo")
plt.axhline(analytic_result*erf(1/np.sqrt(2)), label="1 sigma", c="fuchsia")
plt.xlabel("symmetric integration limits [adimensional]")
plt.legend()
plt.savefig("integral.pdf", format="pdf")
# %%

from scipy.interpolate import interp1d
from scipy.optimize import newton
from scipy.special import erf

spline = interp1d(lims, integrals, kind = "cubic")
real_spline = interp1d(lims, real_integrals, kind = "cubic")

centeredspline = lambda x: spline(x) - erf(1/np.sqrt(2))*analytic_result
real_centeredspline = lambda x: real_spline(x) - erf(1/np.sqrt(2))*analytic_result

im_root = newton(centeredspline, x0=1./2.)
re_root = newton(real_centeredspline, x0=1./2.)
print(im_root)
print(re_root)

#%%

# full_integral = sp.Abs(sp.integrate(exponential, (dy, -sp.oo, sp.oo)))

# x = sp.symbols('x')
# value = 0.69

# x_integral = sp.integrate(exponential, (dy, -x, x))

# ident = sp.Eq(x_integral*sp.conjugate(x_integral), (full_integral * value)**2)


#%%
# sp.solve(ident, x)

#%%
