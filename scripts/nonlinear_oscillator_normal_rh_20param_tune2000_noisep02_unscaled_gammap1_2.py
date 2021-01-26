import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas
import sunode.wrappers.as_theano as sun


from scipy.integrate import ode
alpha  = -0.1
beta= 2
gamma=-2
delta=-0.1
def dX_dt(t, state,par):
    """ Return the growth rate of fox and rabbit populations. """
    alpha,beta,gamma,delta = par
    return np.array([ alpha * state[0]**3 + beta*state[1]**3,
                  gamma*state[0]**3 + delta*state[1]**3])

t = np.linspace(0, 20,  40)              # time
X0 = np.array([2, 0])                    # initials conditions: 10 rabbits and 5 foxes
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

np.random.seed(0)
yobs = X.T + np.random.normal(size=X.T.shape) * 0.02  #np.maximum(X.T + 2*np.random.randn(*X.T.shape),1)
times = t
print(yobs.std(axis=0))
yobs_norm = yobs # / yobs.std(axis=0)


## Do Bayesian Sindy
def nonlinear_oscillator_sunode_library(t, y, p):
    #state = np.array([y.u**3, y.v**3])
    #res = p.pn @ state 
    du_dt = p.pn[0] * y.u**3 + p.pn[1] * y.v**3 + p.pn[4] * y.u + p.pn[6] * y.v + p.pn[8] * y.u**2 + p.pn[10] * y.v**2 + p.pn[12] * y.u*y.v + p.pn[14] * y.u**2 * y.v + p.pn[16] * y.v**2 * y.u + p.pn[18]
    dv_dt = p.pn[2] * y.u**3 + p.pn[3] * y.v**3 + p.pn[5] * y.u + p.pn[7] * y.v + p.pn[9] * y.u**2 + p.pn[11] * y.v**2 + p.pn[13] * y.u*y.v + p.pn[15] * y.u**2 * y.v + p.pn[17] * y.v**2 * y.u + p.pn[19]
    return {'u': du_dt - 1e-5 * y.u**5, 'v' : dv_dt - 1e-5 * y.v**5}


model_sunode = pm.Model()

d = 20

slab_df = 4
slab_scale = 2

with model_sunode:

    #sigma = pm.Lognormal('sigma', mu=-1, sigma=1, shape=2) #  pm.Gamma('sigma',1,1,shape=2) #
    sigma = pm.Gamma('sigma',1,0.1,shape=2) 
    #pn = pm.Laplace('pn', mu=0, b=1, shape=d)
    
    
    l = pm.HalfStudentT('l', nu=1, sigma=1, shape=d)
    tau = pm.HalfStudentT('tau', nu=1, sigma=0.1)
    c2 = pm.InverseGamma('c2', alpha=0.5*slab_df, beta=0.5*slab_df*slab_scale**2)
    
    lt = pm.Deterministic('lt', pm.math.sqrt(c2)*l / pm.math.sqrt(c2 + pm.math.sqr(tau) * pm.math.sqr(l)))
    
    z  = pm.Normal('z', mu=0, sigma=1, shape=d)
    pn = pm.Deterministic('pn', z*tau*lt)
    
    #y0 = pm.Normal('y0',mu=yobs_norm[0,:], sigma=0.01, shape=2)
    #xi = pm.Bernoulli('xi', 0.8, shape=d
    #pnss = pm.Deterministic('pnss', pn * xi)
    y0 = pm.Laplace('y0', mu=0, b=1, shape=2)

    y_hat, _, problem, solver, _, _ = sun.solve_ivp(
        
        y0={
            'u': (y0[0], ()),
            'v': (y0[1], ()),
            },
            params={
                'pn' : (pn,d),
                'tmp': np.zeros(1),  # Theano wants at least one fixed parameter
            },
            rhs=nonlinear_oscillator_sunode_library,
            make_solver='RK45',
            tvals=times,
            t0=times[0],
        )


    import sunode
    lib = sunode._cvodes.lib
    lib.CVodeSetMaxNumSteps(solver._ode, 1200)
    lib.CVodeSetMaxNumStepsB(solver._ode, solver._odeB, 1200)

    uobs = pm.Normal('uobs', mu=y_hat['u'][:], sigma=sigma[0], observed=yobs_norm[:,0])
    vobs = pm.Normal('vobs', mu=y_hat['v'][:], sigma=sigma[1], observed=yobs_norm[:,1])

with model_sunode:

    trace = pm.sample(2000, tune=2000, cores=2,random_seed=2,target_accept=0.9) #,start=start)

    pm.backends.save_trace(trace,'nonlinear_oscillator_normal_rh_20param_tune2000_noisep02_unscaled_gammap1_2.trace',model_sunode)


print(__file__)

