import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas
import sunode.wrappers.as_theano as sun
from scipy.integrate import ode


## Generate Data
def f(t,state,par):
    σ,ρ,β = par
    x,y,z = state
    return np.array([σ*(y-x),
                  x*(ρ-z)-y,
                  x*y-β*z])

# Jacobian
def Df(t,state,par):
    σ,ρ,β = par
    x,y,z = state
    print(np.array([[-σ,σ,0],[ρ-z,-1,-x],[y,x,-β]]))
    return np.array([[-σ,σ,0],[ρ-z,-1,-x],[y,x,-β]])

step = 0.2
n_iter = 100
#t = np.arange(0, 100, step)
t_vals = np.arange(0,n_iter*step,step)
Y = np.empty((3, t_vals.size))
#Y[:,0] = np.array([1.508870,-1.531271,25.46091])

Y[:,0] = np.array([6.7673,6.1253,25.8706])
σ,ρ,β = 10,28,8.0/3.0

#state = odeint(f, y0, t, args=(σ,ρ,β))

r = ode(f).set_integrator('dopri5')
r.set_initial_value(Y[:,0], t_vals[0])
r.set_f_params((σ,ρ,β))

for i,t in enumerate(t_vals):
    if i == 0:
        continue
    r.integrate(t)
    Y[:,i] = r.y
    
np.random.seed(0)
yobs = Y.T  + np.random.normal(size=Y.T.shape) * 0.01 #* np.random.lognormal(mean=0,sigma=0.01,size=Y.T.shape)  #np.maximum(X.T + 2*np.random.randn(*X.T.shape),1)
#times = np.arange(1, len(t_vals) + 1).astype(float)
print(yobs.std(axis=0))
yobs_norm = yobs[40:70,:] #/ yobs.std(axis=0)
#N = len(times)
t_vals = t_vals[40:70]



## Do Bayesian Sindy
def lorenz_sunode_library(t, y, p):
    #np.array([y.x, y.yy])
    dx_dt = p.pn[0] * y.x + p.pn[1] * y.y + p.pn[7] * y.z
    dy_dt = p.pn[2] * y.x + p.pn[3] * y.x * y.z + p.pn[4] * y.y + p.pn[8] * y.z
    dz_dt = p.pn[5] * y.x * y.y + p.pn[6] * y.z + p.pn[9] * y.x
    return {'x': dx_dt - 1e-7 * y.x**3, 'y' : dy_dt - 1e-7 * y.y**3, 'z' : dz_dt - 1e-7 * y.z**3}

model_sunode = pm.Model()

d = 10


with model_sunode:

    sigma = pm.Lognormal('sigma', mu=-3, sigma=1, shape=3)
    
    l = pm.HalfStudentT('l', nu=1, sigma=1, shape=d)
    tau = pm.HalfStudentT('tau', nu=1, sigma=0.1)
    #c2 = pm.InverseGamma('c2', alpha=0.5*slab_df, beta=0.5*slab_df*slab_scale**2)
    
    #lt = (pm.math.sqrt(c2)*l) / pm.math.sqrt(c2 + pm.math.sqr(tau) * pm.math.sqr(l))
    
    z  = pm.Normal('z', mu=0, sigma=10, shape=d)
    pn = pm.Deterministic('pn', z*tau*l)
    #pn = pm.Normal('pn', mu=0, sigma=10, shape=d)
    
    y0 = pm.Normal('y0',mu=yobs_norm[0,:], sigma=0.01, shape=3) #pm.Lognormal('y0', mu=pm.math.log(10), sigma=1, shape=3)

    y_hat, _, problem, solver, _, _  = sun.solve_ivp(
        y0={
            'x': (y0[0], ()),
            'y': (y0[1], ()),
            'z': (y0[2], ()),
            },
            params={
                'pn' : (pn,d),
                'tmp': np.zeros(1),  # Theano wants at least one fixed parameter
            },
            rhs=lorenz_sunode_library,
    make_solver='BDF',
            tvals=t_vals,
            t0=t_vals[0],
        )

    #import pysindy as ps
    #from pysindy.differentiation import SmoothedFiniteDifference
    #sfd = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
    #dx = sfd(inp)

    #dx / step        

        
    xobs = pm.Normal('xobs', mu=y_hat['x'][:], sigma=sigma[0], observed=yobs_norm[:,0])
    yobs = pm.Normal('yobs', mu=y_hat['y'][:], sigma=sigma[1], observed=yobs_norm[:,1])
    zobs = pm.Normal('zobs', mu=y_hat['z'][:], sigma=sigma[2], observed=yobs_norm[:,2])
    
    #xobs = pm.Lognormal('xobs', mu=pm.math.log(y_hat['x'][:]), sigma=sigma[0], observed=yobs_norm[:,0])
    #yobs = pm.Lognormal('yobs', mu=pm.math.log(y_hat['y'][:]), sigma=sigma[1], observed=yobs_norm[:,1])
    #zobs = pm.Lognormal('zobs', mu=pm.math.log(y_hat['z'][:]), sigma=sigma[2], observed=yobs_norm[:,2])
    

with model_sunode:

    start = pm.find_MAP()

    # Initialize parameters with least squares and all other values with MAP
    inp = yobs_norm
    x = inp[:,0]
    y = inp[:,1]
    z = inp[:,2]

    θ = np.array([x,y,z,x*y,x*z]).T

    import pysindy as ps
    from pysindy.differentiation import SmoothedFiniteDifference
    sfd = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
    dx = sfd(inp)

    guess = np.linalg.lstsq(θ,dx)[0] / (t_vals[1] - t_vals[0])
    
    print('Initialization')
    print(guess)

    #start['pnss'] = start['pn']
    '''start['tau'] = 0.1
    start['sigma'] = 0.1 * np.ones(start['sigma'].shape)
    start['lt'] = (np.sqrt(start['c2'])*start['l']) / np.sqrt(start['c2'] + start['tau']**2 * start['l']**2)
    start['tau_log__'] = np.log(start['tau'])
    start['pn'] = guess.T
    start['z'] = start['pn'] / start['tau'] / start['lt']
    start['sigma_log__'] = np.log(start['sigma'])
    start['z_log__'] = np.log(start['z'])
    start['c2_log__'] = np.log(start['c2'])
    start['l_log__'] = np.log(start['l'])
    start['lt_log__'] = np.log(start['lt'])
    start['y0'] = yobs_norm[0,:]
    start['y0_log__'] = np.log(start['y0'])'''


    start['tau'] = 0.1
    #start['c2'] = 5
    start['l'] = np.ones(start['l'].shape) * 10
    #start['sigma'] = 0.1 * np.ones(start['sigma'].shape)
    #start['tau_log__'] = np.log(start['tau'])
    #start['pn'] = guess.Th

    start['pn'] = np.array([-σ, σ, ρ, -1, -1, 1, -β, 0.1, 0.1, 0.1])
    start['z'] = start['pn'] / start['tau'] / start['l']
    #start['z'] = np.sign(guess).flatten() 
    #start['lt'] = (np.sqrt(start['c2'])*start['l']) / np.sqrt(start['c2'] + start['tau']**2 * start['l']**2)
    #start['z'] = start['pn'] / start['tau'] / start['l'] # (np.sqrt(start['c2'])*start['l']) / np.sqrt(start['c2'] + start['tau']**2 * start['l']**2)
    #start['c2_log__'] = np.log(start['c2'])
    start['l_log__'] = np.log(start['l'])
    #start['lt_log__'] = np.log(start['lt']) 
    start['tau_log__'] = np.log(start['tau'])
    start['sigma'] = 0.01 * np.ones(3)
    start['sigma_log__'] = np.log(start['sigma'])
    #start['y0'] = yobs_norm[0,:]
    start['y0'] = yobs_norm[0,:]
    #start['y0_log__'] = np.log(start['y0'])
    #start['sigma_log__'] = np.log(start['sigma'])
    print(start)




    trace = pm.sample(2000, tune=1000, cores=2, random_seed=0, target_accept=0.95, start=start)



    #trace = pm.sample(1000, tune=500, cores=2, random_seed=0, target_accept=0.99)

    pm.backends.save_trace(trace,'lorenz_2lobe_h_10param_attempt5' + '.trace',model_sunode)

print(__file__)
print('done')