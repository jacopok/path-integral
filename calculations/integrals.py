# %%

import sympy as sp
# sp.init_printing()

dy = sp.symbols('dy', real=True)

# %% 

exponential = sp.exp(sp.I * 4 * dy ** 2)

analytic_result = float(sp.Abs(sp.integrate(exponential, (dy, -sp.oo, sp.oo)).n()))

#%%

import numpy as np
lims = np.arange(0, 5, 0.02)
integrals = []

for lim in lims:
    c_value = sp.integrate(exponential, (dy, -lim, lim))
    r_value = sp.Abs(c_value.n())
    integrals.append(float(r_value))

#%%

import matplotlib.pyplot as plt
plt.style.use('seaborn')

plt.plot(lims, integrals, label = "Integral")
plt.axhline(analytic_result, c="g", label="sqrt(pi)/2")
plt.xlabel("symmetric integration limits [adimensional: deltaX / (hbar * T / m)]")
plt.legend()
plt.savefig("integral.pdf", format="pdf")
#%%

full_integral = sp.Abs(sp.integrate(exponential, (dy, -sp.oo, sp.oo)))

x = sp.symbols('x')
value = 0.69

ident = sp.Eq(sp.Abs(sp.integrate(exponential, (dy, -x, x))), full_integral * value)


#%%
sp.solveset(ident, x)

#%%
