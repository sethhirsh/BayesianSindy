functions {
  real[] dz_dt(real t,       // time
               real[] z,     // system state {prey, predator}
               real[] theta, // parameters
               real[] x_r,   // unused data
               int[] x_i) {
    real u = z[1];
    real v = z[2];

    real du_dt = theta[1] * u + theta[3] * v + theta[5] * u * v + theta[7] * u * u + theta[9] * v * v - 1e-3 * u * u * u;
    real dv_dt = theta[2] * u + theta[4] * v + theta[6] * u * v + theta[8] + u * u + theta[10] * v * v - 1e-3 * v * v * v;

    return { du_dt, dv_dt };
  }
}
data {
  int<lower = 0> N;          // number of measurement times
  real ts[N];                // measurement times > 0
  real y_init[2];            // initial measured populations
  real<lower = 0> y[N, 2];   // measured populations
  real slab_df;
  real slab_scale;
  int<lower = 0> d;
}
parameters {
  vector <lower = 0>[d] lambda;
  real<lower = 0> z_init[2];  // initial population
  vector <lower = 0>[2] sigma;   // measurement errors
  real<lower=0> tau;
  vector[d] my_z;
  real <lower=0> caux;
  
}
transformed parameters {
  vector<lower=0>[d] lambda_tilde;
  real theta[d];
  real z[N,2];
  lambda_tilde = (sqrt(caux) * lambda) ./ sqrt(caux + tau^2*square(lambda)); 
  for (k in 1:d){
     theta[k] = my_z[k] * lambda_tilde[k] * tau;}
  z = integrate_ode_bdf(dz_dt, z_init, 0, ts, theta,
                         rep_array(0.0, 0), rep_array(0, 0),
                         1e-5, 1e-3, 2e3);
}
model {
  lambda ~ student_t(1.0, 0.0, 1.0);
  tau ~ student_t(1.0, 0.0, 0.01);
  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df * square(slab_scale));
  my_z ~ normal(0.0, 1.0);
  sigma ~ lognormal(-1, 0.1);
  z_init ~ lognormal(log(1), 1);
  for (k in 1:2) {
    y[ , k] ~ lognormal(log(z[, k]), sigma[k]);
  }
}
generated quantities {
  real y_init_rep[2];
  real y_rep[N, 2];
  for (k in 1:2) {
    y_init_rep[k] = lognormal_rng(log(z_init[k]), sigma[k]);
    for (n in 1:N)
      y_rep[n, k] = lognormal_rng(log(z[n, k]), sigma[k]);
  }
}