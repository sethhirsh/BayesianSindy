import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas
import sunode.wrappers.as_theano as sun

def predator_prey_sunode_library(t, y, p):
    du_dt = p.p0 * y.u + p.pn0 * y.u * y.u + p.pn2 * y.u * y.v
    dv_dt = p.p1 * y.v + p.pn1 * y.v * y.v + p.pn3 * y.u * y.v
    return {'u': du_dt, 'v' : dv_dt}


def compute_trace(σ,β):
    print(σ,β)

    model_sunode = pm.Model()

    with model_sunode:

        #sigma = pm.Lognormal('sigma', mu=-1, sigma=0.1, shape=2)
        sigma = pm.Lognormal('sigma', mu=-1, sigma=σ, shape=2)
        p0 = pm.Normal('p0', mu=0, sigma=1.0)
        p1 = pm.Normal('p1', mu=0, sigma=1.0)

        # pn = pm.Normal('pn', mu=0, sigma=0.1, shape=4)
        pn = pm.Laplace('pn', mu=0, b=0.1, shape=4)

        r  = pm.Beta('r', 1, β)
        xi = pm.Bernoulli('xi', r, shape=4)

        #xi = pm.Bernoulli('xi', 0.8, shape=4)

        pnss = pm.Deterministic('pnss', pn * xi)

        y0 = pm.Lognormal('y0', mu=pm.math.log(10), sigma=1, shape=2)

        y_hat = sun.solve_ivp(
            y0={
                'u': (y0[0], ()),
                'v': (y0[1], ()),
                },
                params={
                    'p0': (p0, ()),
                    'p1': (p1, ()),
                    'pn0': (pnss[0], ()),
                    'pn1': (pnss[1], ()),
                    'pn2': (pnss[2], ()),
                    'pn3': (pnss[3], ()),
                    'tmp': np.zeros(1),  # Theano wants at least one fixed parameter
                },
                rhs=predator_prey_sunode_library,
                tvals=times,
                t0=times[0],
            )[0]

        uobs = pm.Lognormal('uobs', mu=pm.math.log(y_hat['u'][:]), sigma=sigma[0], observed=yobs[:,0])
        vobs = pm.Lognormal('vobs', mu=pm.math.log(y_hat['v'][:]), sigma=sigma[1], observed=yobs[:,1])

    with model_sunode:
        trace = pm.sample(1000, tune=1000, cores=2, step_kwargs={'nuts':{'target_accept':0.95}})

        pm.backends.save_trace(trace,'traces/ppsunodelibrary' + 'sigma=' + str(σ) + 'beta=' + str(β) + '.trace',model_sunode)
    print('done')

df = pandas.read_csv('hudson-bay-linx-hare.csv',header=1)

year = df['Year']
lynx = df['Lynx']
hare = df['Hare']

times = np.array(year,dtype=float)
yobs = np.array([hare,lynx]).T

    
for σ in [0.1,]:
    for β in [0.5,2,4,8,16]:
        compute_trace(σ,β)
