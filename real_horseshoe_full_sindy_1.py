import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas
import sunode.wrappers.as_theano as sun

def predator_prey_sunode_library(t, y, p):
    du_dt = p.pnone0 + p.pn00 * y.u + p.pn0 * y.v + p.pn2 * y.u * y.v + p.pn4 * y.u**2 + p.pn6* y.v**2 - 1e-5 * y.u**3
    dv_dt = p.pnone1 + p.pn11 * y.v + p.pn1 * y.u + p.pn3 * y.u * y.v + p.pn5 * y.u**2 + p.pn7*y.v**2 - 1e-5 * y.v**3
    return {'u': du_dt, 'v' : dv_dt}


df = pandas.read_csv('hudson-bay-linx-hare.csv',header=1)

year = df['Year']
lynx = df['Lynx']
hare = df['Hare']

times = np.array(year,dtype=float)
yobs = np.array([hare,lynx]).T 


model_sunode = pm.Model()

with model_sunode:

    sigma = pm.Lognormal('sigma', mu=-1, sigma=0.1, shape=2)
    #sigma = pm.Lognormal('sigma', mu=0., sigma=1.0, shape=2)
    pnlarge = pm.Laplace('pnlarge', mu=0, b=1,shape=4)
    #p1 = pm.Normal('p1', mu=0, sigma=1.0)

    pnone = pm.Laplace('pnone', mu=0, b=1, shape=2)

    #p2 = pm.Normal('p2', mu=0, sigma=1.0)
    #p3 = pm.Normal('p3', mu=0, sigma=1.0)

    #pn = pm.Normal('pn', mu=0, sigma=0.1, shape=4)
    pn = pm.Laplace('pn', mu=0, b=0.1, shape=6)

   # r  = pm.Beta('r', 1, beta)
   # xi = pm.Bernoulli('xi', r, shape=4)

    xi = pm.Bernoulli('xi', 0.8, shape=6)
    xione = pm.Bernoulli('xione', 0.8, shape=2)

    xilarge = pm.Bernoulli('xilarge', 0.8,shape=4)
    #xip1 = pm.Bernoulli('xip1', 0.8)
    pnsslarge = pm.Deterministic('pnsslarge',pnlarge* xilarge)
    #pn1 = pm.Deterministic('pn1',p1 * xip1)
    pnssone = pm.Deterministic('pnssone',pnone* xione)

    pnss = pm.Deterministic('pnss', pn * xi)

    y0 = pm.Lognormal('y0', mu=pm.math.log(10), sigma=1, shape=2)

    y_hat = sun.solve_ivp(
        y0={
            'u': (y0[0], ()),
            'v': (y0[1], ()),
            },
            params={
                'pnone0' : (pnssone[0], ()),
                'pnone1' : (pnssone[1], ()),
                'pn00': (pnsslarge[0], ()),
                'pn11': (pnsslarge[1], ()),
                'pn0': (pnsslarge[2], ()),
                'pn1': (pnsslarge[3], ()),
                'pn2': (pnss[0], ()),
                'pn3': (pnss[1], ()),
                'pn4': (pnss[2], ()),
                'pn5': (pnss[3], ()),
                'pn6': (pnss[4], ()),
                'pn7': (pnss[5], ()),
                'tmp': np.zeros(1),  # Theano wants at least one fixed parameter
            },
            rhs=predator_prey_sunode_library,
    make_solver='BDF',
            tvals=times,
            t0=times[0],
        )[0]

    uobs = pm.Lognormal('uobs', mu=pm.math.log(y_hat['u'][:]), sigma=sigma[0], observed=yobs[:,0])
    vobs = pm.Lognormal('vobs', mu=pm.math.log(y_hat['v'][:]), sigma=sigma[1], observed=yobs[:,1])

with model_sunode:

    start = pm.find_MAP()
    #start['p0'] = 0.544
    #start['p1'] = -0.988
    #start['pn0'] = start['p0']
    #start['pn1'] = start['p1']

    #start['pn0'] = -0.292
    #start['pn1'] = 0.056
    #start['pn2'] = -0.010
    #start['pn3'] = 0.007
    #start['pn4'] = -0.003
    #start['pn5'] = 0.003
    #start['pn6'] = -0.001
    #start['pn7'] = 0.012
    start['pnone'] = [-3.726,3.388]
    start['pnssone'] = start['pnone']
    start['pnlarge'] = np.array([0.677,-1.140,-0.124,-0.066])
    start['pnsslarge'] = start['pnlarge']

    start['pn'] = np.array([-0.010,0.008,-0.005,0.004,-0.004,0.014])
    start['pnss'] = start['pn']
    start['xilarge'] = [1,1,1,1]
    start['xione'] = [1,1]
    start['xi'] = [1,1,1,1,1,1]
    start['y0'] = yobs[0,:]

    trace = pm.sample(1000, tune=1000, cores=2, start=start, step_kwargs={'nuts':{'target_accept':0.95}})

    pm.backends.save_trace(trace,'real_reg_minus_5_full_sindy_9' + '.trace',model_sunode)

    print('ahwejhefwggfi9')
print('done')


    

