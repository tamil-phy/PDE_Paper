import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import CONFIG

mu = 1.0

def vanderpol(X, t):
    x = X[0]
    y = X[1]
    dxdt = mu * (x - 1./3.*x**3 - y)
    dydt = x / mu
    return [dxdt, dydt]

X0 = [1, 2]
t  = np.linspace(0, 20, 250000)

sol = odeint(vanderpol, X0, t)

filepath = __file__.replace('.py', '.pkl')
print('dumping data to {}'.format(filepath))
pickle.dump((t, sol),
             open('{}/{}'.format(CONFIG.DATA_DIR, filepath), 'wb'))

x = sol[:, 0]
y = sol[:, 1]


plt.plot(t,x, t, y)
plt.xlabel('t')
plt.legend(('x', 'y'))


# phase portrait
plt.figure()
plt.plot(x, y)
plt.plot(x[0], y[0], 'ro')
plt.xlabel('x')
plt.ylabel('y')                 
