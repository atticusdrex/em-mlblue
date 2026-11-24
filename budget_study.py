# %% importing everything 
from util import * 
from matplotlib.pyplot import * 

# Global Variable Settings
N_SAMPS    = 10
SAMP_COEFS = jnp.logspace(jnp.log10(10), 4, num = N_SAMPS).astype(int) 
N_REPS     = 1000
KEYS       = jrand.split(jrand.PRNGKey(42), N_REPS) 
COND       = 100.0 
F          = set([0,1,2,3]) # defining the set of models 
G          = nonempty_powerset(F) # returning a set of frozensets
G          = [set([0,1,2,3]), set([0,1,2]), set([0,1]), set([0])] # nested set
L, K       = len(F), len(G) # extracting the number of levels of fidelity and the 
R          = [jnp.column_stack([jnp.eye(L)[:,j] for j in Si]).T for Si in G] 
GAMMA      = jnp.eye(L) * 1e-6
GAMMA_SQRT = jnp.diag(jnp.sqrt(jnp.diag(GAMMA)))
SOLVER_TOL = 1e-6 
MAX_ITER   = 1000 
SIGMA_0    = 1e-2*jnp.eye(L)

# %%
mu_mean, mu_conf = [], []
sigma_mean, sigma_conf = [], []
mlblue_mean, mlblue_conf = [], [] 

mlblue5_mean, mlblue5_conf = [], [] 
sigma5_mean, sigma5_conf = [], []

for i, samp_coef in tqdm(enumerate(SAMP_COEFS), total = N_SAMPS):
    ns = samp_coef * jnp.ones(shape=(4,)).astype(int)

    def run_rep(key): 
        # defining the target covariance 
        V, _ = jnp.linalg.qr(jrand.normal(key, shape = (L,L)), mode = 'complete')
        Lambda = jnp.diag(jnp.linspace(1.0/COND, 1.0, L))
        Lambda_sqrt = jnp.diag(jnp.sqrt(jnp.diag(Lambda)))
        Sigma = V @ Lambda @ V.T 

        # collecting data for a given level of fidelity
        data = []
        keys = jrand.split(key, K) 
        for k, group in enumerate(G): 
            samp_key, noise_key = jrand.split(keys[k], 2)
            samp = R[k] @ (V @ Lambda_sqrt @ jrand.normal(samp_key, shape = (L, int(ns[k])))+ GAMMA_SQRT @ jrand.normal(noise_key, shape = (L, int(ns[k]))))
            data.append(samp) 

        # running the EM algorithm 
        mu_em, Sigma_em, _, _ = em_mlblue(
            data, ns, R, GAMMA, SIGMA_0, tol = SOLVER_TOL, max_iter = MAX_ITER
        )

        # running MLBLUE with perfect covariances 
        mu_mlblue = mlblue_solve(data, ns, R, Sigma, GAMMA)

        # running MLBLUE with 10-sample pilot covariances
        samp_key, noise_key = jrand.split(key, 2) 
        samp = V @ Lambda_sqrt @ jrand.normal(samp_key, shape = (L, 5))
        Sigma5 = matrix_cov(samp.T, samp.T) 
        mu_mlblue5 = mlblue_solve(data, ns, R, Sigma5, GAMMA) 


        return jnp.linalg.norm(mu_em), jnp.linalg.norm(mu_mlblue), jnp.linalg.norm(mu_mlblue5), jnp.linalg.norm(Sigma_em - Sigma, ord=2), jnp.linalg.norm(Sigma5 - Sigma, ord=2)

    mu_errors, mlblue_errors, mlblue5_errors, sigma_errors, sigma5_errors = vmap(run_rep)(KEYS)

    mu_mean.append(jnp.array(mu_errors).mean()) 
    mu_conf.append(jnp.array(mu_errors).std()*2)

    sigma_mean.append(jnp.array(sigma_errors).mean())
    sigma_conf.append(jnp.array(sigma_errors).std()*2) 

    mlblue_mean.append(jnp.array(mlblue_errors).mean()) 
    mlblue_conf.append(jnp.array(mlblue_errors).std()*2)

    mlblue5_mean.append(jnp.array(mlblue5_errors).mean()) 
    mlblue5_conf.append(jnp.array(mlblue5_errors).std()*2)

    sigma5_mean.append(jnp.array(sigma5_errors).mean())
    sigma5_conf.append(jnp.array(sigma5_errors).std()*2)



# %% 
figure(figsize=(8, 3.5), dpi = 400) 

subplot(1,2,1) 
# yscale('log')
xscale('log') 

plot(SAMP_COEFS, mu_mean, color = 'blue', linestyle = 'dashed', marker = 's', label = "MLBLUE (EM) $\\pm 1 \\sigma$")
fill_between(SAMP_COEFS, jnp.array(mu_mean) - jnp.array(mu_conf), jnp.array(mu_mean) + jnp.array(mu_conf), alpha = 0.6, color = 'lightblue')

plot(SAMP_COEFS, mlblue_mean, color = 'orange', linestyle = 'dashed', marker = 's', label = "MLBLUE (exact) $\\pm 1 \\sigma$")
fill_between(SAMP_COEFS, jnp.array(mlblue_mean) - jnp.array(mlblue_conf), jnp.array(mlblue_mean) + jnp.array(mlblue_conf), alpha = 0.6, color = 'orange')

plot(SAMP_COEFS, mlblue5_mean, color = 'green', linestyle = 'dashed', marker = 's', label = "MLBLUE (5 pilots) $\\pm 1 \\sigma$")
fill_between(SAMP_COEFS, jnp.array(mlblue5_mean) - jnp.array(mlblue5_conf), jnp.array(mlblue5_mean) + jnp.array(mlblue5_conf), alpha = 0.1, color = 'green')
ylim(0, 0.9)
ylabel("$||\\hat \\mu - \\mu||_2$")
xlabel("Per-group sample size") 
legend()
title("Mean Error") 

subplot(1,2,2) 
# yscale('log') 
xscale('log') 

plot(SAMP_COEFS, sigma_mean, color = 'blue', linestyle = 'dashed', marker = 's', label = "MLBLUE (EM)")
fill_between(SAMP_COEFS, jnp.array(sigma_mean) - jnp.array(sigma_conf), jnp.array(sigma_mean) + jnp.array(sigma_conf), alpha = 0.6, color = 'lightblue')

plot(SAMP_COEFS, sigma5_mean, color = 'green', linestyle = 'dashed', marker = 's', label = "MLBLUE (5 pilots)")
fill_between(SAMP_COEFS, jnp.array(sigma5_mean) - jnp.array(sigma5_conf), jnp.array(sigma5_mean) + jnp.array(sigma5_conf), alpha = 0.1, color = 'green')

xlabel("Per-group sample size") 
ylabel("$||\\hat \\Sigma - \\Sigma||_2$")
legend()
title("Covariance Error")

tight_layout()

# %% running on non-Gaussian distribution 

mu_mean, mu_conf = [], []
sigma_mean, sigma_conf = [], []
mlblue_mean, mlblue_conf = [], [] 

for i, samp_coef in tqdm(enumerate(SAMP_COEFS), total = N_SAMPS):
    ns = samp_coef * jnp.ones(shape=(4,)).astype(int)

    def run_rep(key): 
        # defining the target covariance 
        V, _ = jnp.linalg.qr(jrand.normal(key, shape = (L,L)), mode = 'complete')
        Lambda = jnp.diag(jnp.linspace(1.0/COND, 1.0, L))
        Lambda_sqrt = jnp.diag(jnp.sqrt(jnp.diag(Lambda)))
        Sigma = V @ Lambda @ V.T 

        # collecting data for a given level of fidelity
        data = []
        keys = jrand.split(key, K) 
        for k, group in enumerate(G): 
            samp_key, noise_key = jrand.split(keys[k], 2)
            samp = R[k] @ (V @ Lambda_sqrt @ (jrand.exponential(samp_key, shape = (L, int(ns[k]))) - 1.0) + GAMMA_SQRT @ jrand.normal(noise_key, shape = (L, int(ns[k]))))
            data.append(samp) 

        # running the EM algorithm 
        mu_em, Sigma_em, _, _ = em_mlblue(
            data, ns, R, GAMMA, SIGMA_0, tol = SOLVER_TOL, max_iter = MAX_ITER
        )

        # running MLBLUE with perfect covariances 
        mu_mlblue = mlblue_solve(data, ns, R, Sigma, GAMMA)

        return mu_em[-1], mu_mlblue[-1], (Sigma_em - Sigma)[-1,-1]

    mu_errors, mlblue_errors, sigma_errors = vmap(run_rep)(KEYS)

    mu_mean.append(jnp.array(mu_errors).mean()) 
    mu_conf.append(jnp.array(mu_errors).std()*1)

    sigma_mean.append(jnp.array(sigma_errors).mean())
    sigma_conf.append(jnp.array(sigma_errors).std()*1) 

    mlblue_mean.append(jnp.array(mlblue_errors).mean()) 
    mlblue_conf.append(jnp.array(mlblue_errors).std()*1)

    

# %% 
figure(figsize=(8, 3.5), dpi = 200) 

subplot(1,2,1) 
xscale('log')
suptitle("The MLBLUE with EM has negligible bias on non-symmetric non-Gaussian distributions")
plot(SAMP_COEFS, mu_mean, color = 'blue', linestyle = 'dashed', marker = 's', label = "EM MLBLUE $\\pm 1 \\sigma$")
fill_between(SAMP_COEFS, jnp.array(mu_mean) - jnp.array(mu_conf), jnp.array(mu_mean) + jnp.array(mu_conf), alpha = 0.6, color = 'lightblue')

plot(SAMP_COEFS, mlblue_mean, color = 'orange', linestyle = 'dashed', marker = 's', label = "MLBLUE $\\pm 1 \\sigma$")
fill_between(SAMP_COEFS, jnp.array(mlblue_mean) - jnp.array(mlblue_conf), jnp.array(mlblue_mean) + jnp.array(mlblue_conf), alpha = 0.6, color = 'orange')

plot(SAMP_COEFS, jnp.zeros_like(jnp.array(SAMP_COEFS)), linestyle = 'dotted', color = 'black')
ylabel("$\\hat \\mu_4 - \\mu_4$")
xlabel("Per-group sample size") 
legend()
title("High-Fidelity Mean Error") 

subplot(1,2,2) 
xscale('log')

plot(SAMP_COEFS, sigma_mean, color = 'blue', linestyle = 'dashed', marker = 's', label = "EM MLBLUE")
fill_between(SAMP_COEFS, jnp.array(sigma_mean) - jnp.array(sigma_conf), jnp.array(sigma_mean) + jnp.array(sigma_conf), alpha = 0.6, color = 'lightblue')
plot(SAMP_COEFS, jnp.zeros_like(jnp.array(SAMP_COEFS)), linestyle = 'dotted', color = 'black')
xlabel("Per-group sample size") 
ylabel("$\\hat \\Sigma_{44} - \\Sigma_{44}$")
legend()
title("High-Fidelity Variance Error")

tight_layout()
# %%
