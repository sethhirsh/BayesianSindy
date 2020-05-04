import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas
import sunode.wrappers.as_theano as sun

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas
import sunode.wrappers.as_theano as sun


def predator_prey_sunode_library(t, y, p):
    du_dt = p.par[0] * y.y[0] + p.par[2] * y.y[0] * y.y[0] + p.par[3] * y.y[0] * y.y[1] - 1e-5 * y.y[0]**3
    dv_dt = p.par[1] * y.y[1] + p.par[4] * y.y[1] * y.y[1] + p.par[5] * y.y[0] * y.y[1] - 1e-5 * y.y[1]**3
    return {'y': [du_dt,dv_dt]} #, 'v' : dv_dt}


from scipy.integrate import ode
alpha  = 1
beta=0.1
gamma=1.5
delta=0.75 * 0.1
def dX_dt(t, state,par):
    """ Return the growth rate of fox and rabbit populations. """
    alpha,beta,gamma,delta = par
    return np.array([ alpha*state[0] -   beta*state[0]*state[1],
                  -gamma*state[1] + delta*state[0]*state[1]])

t = np.linspace(0, 15,  100)              # time
X0 = np.array([10, 5])                    # initials conditions: 10 rabbits and 5 foxes
r = ode(dX_dt).set_integrator('dopri5')
r.set_initial_value(X0, t[0])
r.set_f_params((alpha,beta,gamma,delta))
X = np.zeros((len(X0),len(t)))
X[:,0] = X0
for i, _t in enumerate(t):
    if i == 0:
        continue
    r.integrate(_t)
    X[:, i] = r.y

yobs = X.T #* np.random.lognormal(mean=-1,sigma=0.1,size=X.T.shape)  #np.maximum(X.T + 2*np.random.randn(*X.T.shape),1)
times = t


model_sunode = pm.Model()

with model_sunode:

    sigma = pm.Lognormal('sigma', mu=-1, sigma=0.1, shape=2)
    #sigma = pm.Lognormal('sigma', mu=0., sigma=1.0, shape=2)
    p0 = pm.Normal('p0', mu=0, sigma=1.0,shape=2)
    #p1 = pm.Normal('p1', mu=0, sigma=1.0)

    #pn = pm.Normal('pn', mu=0, sigma=0.1, shape=4)
    pn = pm.Laplace('pn', mu=0, b=0.1, shape=4)

   # r  = pm.Beta('r', 1, beta)
   # xi = pm.Bernoulli('xi', r, shape=4)

    xi = pm.Bernoulli('xi', 0.8, shape=4)

    pnss = pm.Deterministic('pnss', pn * xi)

    y0 = pm.Lognormal('y0', mu=pm.math.log(10), sigma=1, shape=2)
    
    par = pm.math.concatenate((p0,pn))

    y_hat = sun.solve_ivp(
        y0={
            'y': (y0, (2)),
           #'v': (y0[1], ()),
            },
            params={
                #'p0': (p0, ()),
                'par':(par,(6)),
                #'p1': (p1, ()),
                #'pnss': (pnss, (4)),
                'tmp': np.zeros(1),  # Theano wants at least one fixed parameter
            },
            rhs=predator_prey_sunode_library,
    make_solver='BDF',
            tvals=times,
            t0=times[0],
        )[0]

    uobs = pm.Lognormal('uobs', mu=pm.math.log(y_hat['y'][:]), sigma=sigma, observed=yobs[:],shape=2)
with model_sunode:
    trace = pm.sample(1000, tune=1000, cores=2, step_kwargs={'nuts':{'target_accept':0.95}})

    pm.backends.save_trace(trace,'./synthetic/synthetic_reg_minus_5_vec_2' + '.trace',model_sunode)
print('done')


    

