# %% 
from elyza.util.imports import * 
from matplotlib.pyplot import * 
from util import nonempty_powerset 

# GLOBAL VARS 
F = set([0,1,2,3]) # defining the set of models 
G = nonempty_powerset(F) # returning a set of frozensets
G = [set([0,1,2,3]), set([0,1,2]), set([0,1]), set([0])] # nested set
L, K = len(F), len(G) 
R = [jnp.column_stack([jnp.eye(L)[:,j] for j in Si]).T for Si in G] # restriction matrices
vouter = lambda a: a[:, None] * a[None, :] # quick helper function for outer product

# %% sampling a random covariance matrix w/ a specific condition number
key = jrand.PRNGKey(58) # pseudorandom number key 
cond = 2.0 # condition number 
SAMP_COEF = 16
GAMMA = jnp.eye(L) * 1e-6
GAMMA_SQRT = jnp.diag(jnp.sqrt(jnp.diag(GAMMA)))
V, _ = jnp.linalg.qr(jrand.normal(key, shape = (L,L)), mode = 'complete')
Lambda = jnp.diag(jnp.linspace(1.0/cond, 1.0, L))
Lambda_sqrt = jnp.diag(jnp.sqrt(jnp.diag(Lambda)))
Sigma = V @ Lambda @ V.T 
ns = SAMP_COEF * 2**jnp.array([0,1,2,3]) # computing sample allocations 
ns = SAMP_COEF * jnp.ones(shape=(4,)).astype(int)

# collecting data for a given level of fidelity
data = [] # initializing the data 
keys = jrand.split(key, K) 
for i, group in enumerate(G): 
    samp_key, noise_key = jrand.split(keys[i], 2)
    samp = R[i] @ (V @ Lambda_sqrt @ jrand.normal(samp_key, shape = (L, ns[i]))+ GAMMA_SQRT @ jrand.normal(noise_key, shape = (L, ns[i])))
    data.append(samp) 

# performing the EM iteration for mu and sigma 
from jax.scipy.linalg import cho_factor

# Sufficient statistics per group, computed once. Every sum over j in the
# manuscript only needs these, so each iteration costs O(K L^3), not O(N).
S1 = [Y.sum(axis = 1) for Y in data] # sum_j y_ij
S2 = [Y @ Y.T for Y in data]         # sum_j y_ij y_ij^T
N = int(jnp.sum(ns))

def _group_terms(Sigma_t):
    """Cholesky factor of C_i = R_i (Sigma_t + Gamma) R_i^T for every group."""
    return [cho_factor(R[i] @ (Sigma_t + GAMMA) @ R[i].T, lower = True) for i in range(K)]

def _scatter(i, mu):
    """sum_j (y_ij - R_i mu)(y_ij - R_i mu)^T from the sufficient statistics."""
    r = R[i] @ mu
    return S2[i] - jnp.outer(S1[i], r) - jnp.outer(r, S1[i]) + ns[i] * jnp.outer(r, r)

@jit
def em_step(Sigma_t):
    C = _group_terms(Sigma_t)

    # GLS / BLUE fixed point of the mean update:
    # mu_hat = [sum_i n_i R_i^T C_i^{-1} R_i]^{-1} [sum_i R_i^T C_i^{-1} sum_j y_ij]
    Psi = sum(ns[i] * R[i].T @ cho_solve(C[i], R[i]) for i in range(K))
    gam = sum(R[i].T @ cho_solve(C[i], S1[i]) for i in range(K))
    mu_hat = jnp.linalg.solve(Psi, gam)

    # Sigma^{t+1} = (1/N) sum_i [ n_i V_i + sum_j (m_ij - mu_hat)(m_ij - mu_hat)^T ]
    # with K_i = Sigma_t R_i^T C_i^{-1}, m_ij - mu_hat = K_i (y_ij - R_i mu_hat),
    # V_i = Sigma_t - K_i R_i Sigma_t.
    Sigma_next = jnp.zeros((L, L))
    for i in range(K):
        Ki = cho_solve(C[i], R[i] @ Sigma_t).T
        Vi = Sigma_t - Ki @ R[i] @ Sigma_t
        Sigma_next += ns[i] * Vi + Ki @ _scatter(i, mu_hat) @ Ki.T
    Sigma_next = Sigma_next / N
    return mu_hat, 0.5 * (Sigma_next + Sigma_next.T) # symmetrize round-off

@jit
def log_marginal_likelihood(mu, Sigma_t):
    """Observed-data log-likelihood; EM must never decrease this."""
    C = _group_terms(Sigma_t)
    ll = 0.0
    for i in range(K):
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(C[i][0])))
        quad = jnp.trace(cho_solve(C[i], _scatter(i, mu)))
        ll += -0.5 * (ns[i] * (logdet + R[i].shape[0] * jnp.log(2 * pi)) + quad)
    return ll

# initializing with a guess (must be SPD)
Sigma_t = jnp.eye(L)*1e-2
tol, max_iter = 1e-13, 1000
for it in range(max_iter):
    mu_t, Sigma_next = em_step(Sigma_t)
    rel_change = jnp.linalg.norm(Sigma_next - Sigma_t) / jnp.linalg.norm(Sigma_t)
    Sigma_t = Sigma_next
    if rel_change < tol:
        break

C = _group_terms(Sigma)
Psi = sum(ns[i] * R[i].T @ cho_solve(C[i], R[i]) for i in range(K))
gam = sum(R[i].T @ cho_solve(C[i], S1[i]) for i in range(K))
mu_mlblue = jnp.linalg.solve(Psi, gam)

mu_error = mu_t - mu_mlblue

print(f"EM stopped after {it + 1} iterations (rel. change {rel_change:.2e})")
print("log marginal likelihood:", log_marginal_likelihood(mu_t, Sigma_t))

# 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

# 1. Compute shared min/max across both matrices
vmin = min(Sigma_t.min(), Sigma.min())
vmax = max(Sigma_t.max(), Sigma.max())
norm = Normalize(vmin=vmin, vmax=vmax)

# 2. Create subplots with constrained_layout to prevent overlapping labels
fig, axes = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True)

# 3. Plot both matrices using the shared norm
im0 = axes[0].imshow(Sigma_t, norm=norm)
axes[0].set_title(r"Estimated ($\Sigma_t$)")

im1 = axes[1].imshow(Sigma, norm=norm)
axes[1].set_title(r"True ($\Sigma$)")

# 4. Add a single shared colorbar and an overall title
fig.colorbar(im1, ax=axes[1], shrink=0.8)

e_max = vmax - vmin
im2 = axes[2].imshow(Sigma_t - Sigma, norm=Normalize(vmin = -e_max, vmax = e_max), cmap = 'seismic')
axes[2].set_title(r"Absolute Error ($\Sigma_t - \Sigma$)")
fig.suptitle(f"$\mu_t$ Error: [{mu_error[0]:.4f}, {mu_error[1]:.4f}, {mu_error[2]:.4f}, {mu_error[3]:.4f}]")

fig.colorbar(im2, ax=axes[2], shrink=0.8)
plt.show()
# %%
