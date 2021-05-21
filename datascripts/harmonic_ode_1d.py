import pickle
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('bmh')

import CONFIG

class HarmonicOdeSolver:
    def __init__(self, dt, x0, xd0, omega_squared):
        "Inits the solver."
        self.dt = dt
        self.dt_squared = dt**2
        self.t = dt
        self.omega_squared = omega_squared
        self.x0 = x0
        self.xd0 = xd0
        self.x = [xd0 * dt + x0, x0]
        
    def step(self):
        "Steps the solver."
        xt, xtm1 = self.x
        xtp1 = (2 - self.omega_squared * self.dt_squared) * xt - xtm1
        self.x = (xtp1, xt)
        self.t += self.dt
        
    def step_until(self, tmax, snapshot_dt):
        "Steps the solver until a given time, returns snapshots."
        ts = [self.t]
        vals = [self.x[0]]
        niter = max(1, int(snapshot_dt // self.dt))
        while self.t < tmax:
            for _ in range(niter):
                self.step()
            vals.append(self.x[0])
            ts.append(self.t)
        return np.array(ts), np.array(vals)



if __name__ == '__main__':
    solver = HarmonicOdeSolver(dt  = 0.0001, 
                               x0  = 1,
                               xd0 = 0,
                               
                               omega_squared = 4)
    
    tmax, snapshot_dt = 10, 0.00015  #why 0.15?
    
    ts, vals = solver.step_until(tmax, snapshot_dt)

    filepath = __file__.replace('.py', '.pkl')

    pickle.dump((ts, vals),
                open('{}/{}'.format(CONFIG.DATA_DIR, filepath), 'wb'))
    
    plt.plot(ts, vals);
    plt.show()
