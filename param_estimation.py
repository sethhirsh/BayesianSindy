import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas
import sunode.wrappers.as_theano as sun

df = pandas.read_csv('hudson-bay-linx-hare.csv',header=1)

year = df['Year']
lynx = df['Lynx']
hare = df['Hare']

times = np.array(year,dtype=float)
yobs = np.array([hare,lynx]).T

def predator_prey_sunode(t, y, p):
    du_dt = (p.alpha - p.beta * y.v) * y.u
    dv_dt = (-p.gamma + p.delta * y.u) * y.v
    return {'u': du_dt, 'v' : dv_dt}
    
model_sunode = pm.Model()

with model_sunode:
    
    sigma = pm.Lognormal('sigma', mu=-1, sigma=1, shape=2)
    alpha = pm.Normal('alpha', mu=1, sigma=0.5)
    gamma = pm.Normal('gamma', mu=1, sigma=0.5)
    beta  = pm.Normal('beta', mu=0.05, sigma=0.05)
    delta = pm.Normal('delta', mu=0.05, sigma=0.05)
    y0    = pm.Lognormal('y0', mu=pm.math.log(10), sigma=1, shape=2)
    
    y_hat = sun.solve_ivp(
        y0={
            'u': (y0[0], ()),
            'v': (y0[1], ()),
            },
            params={
                'alpha': (alpha, ()),
                'beta':  (beta, ()),
                'gamma': (gamma, ()),
                'delta': (delta, ()),
                'tmp': np.zeros(1),  # Theano wants at least one fixed parameter
            },
            rhs=predator_prey_sunode,
            tvals=times,
            t0=times[0],
        )[0]
    
    uobs = pm.Lognormal('uobs', mu=pm.math.log(y_hat['u'][:]), sigma=sigma[0], observed=yobs[:,0])
    vobs = pm.Lognormal('vobs', mu=pm.math.log(y_hat['v'][:]), sigma=sigma[1], observed=yobs[:,1])
    
with model_sunode:
    trace = pm.sample(1000, tune=500, cores=2, target_accept=0.9, init='adapt_diag')
    pm.save_trace(trace,'param_est.trace')