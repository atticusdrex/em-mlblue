from itertools import chain, combinations
from elyza.util.imports import *  # jnp, jit, cho_solve, pi; also enables 64-bit JAX
from jax import lax
from jax.scipy.linalg import cho_factor
from elyza.util.helpers import matrix_cov

# function for computing powersets excluding the empty set
def nonempty_powerset(input_set):
    s = list(input_set)
    return [set(combo) for combo in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))][1:]


def restriction_matrices(G, L):
    """R_i selects the rows of model group G[i] out of the L models."""
    return [jnp.eye(L)[jnp.array(sorted(Si))] for Si in G]


# ---------------------------------------------------------------------------
# Internal helpers. Everything is passed explicitly (no globals), and the
# loops over groups are plain Python loops, so they unroll at trace time.
# ---------------------------------------------------------------------------

def _sufficient_stats(data):
    """S1_i = sum_j y_ij and S2_i = sum_j y_ij y_ij^T for each group (data[i] is |G_i| x n_i)."""
    S1 = [Y.sum(axis = 1) for Y in data]
    S2 = [Y @ Y.T for Y in data]
    return S1, S2


def _group_terms(Sigma, Gamma, R):
    """Cholesky factor of C_i = R_i (Sigma + Gamma) R_i^T for every group."""
    return [cho_factor(Ri @ (Sigma + Gamma) @ Ri.T, lower = True) for Ri in R]


def _scatter(i, mu, ns, R, S1, S2):
    """sum_j (y_ij - R_i mu)(y_ij - R_i mu)^T from the sufficient statistics."""
    r = R[i] @ mu
    return S2[i] - jnp.outer(S1[i], r) - jnp.outer(r, S1[i]) + ns[i] * jnp.outer(r, r)


def _mlblue_mean(C, ns, R, S1):
    """mu = [sum_i n_i R_i^T C_i^{-1} R_i]^{-1} [sum_i R_i^T C_i^{-1} S1_i]."""
    Psi = sum(ns[i] * R[i].T @ cho_solve(C[i], R[i]) for i in range(len(R)))
    gam = sum(R[i].T @ cho_solve(C[i], S1[i]) for i in range(len(R)))
    return jnp.linalg.solve(Psi, gam)


def _em_step(Sigma_t, ns, R, Gamma, S1, S2):
    """One EM update. Returns (mu_hat, Sigma_next)."""
    C = _group_terms(Sigma_t, Gamma, R)

    # GLS / BLUE fixed point of the mean update
    mu_hat = _mlblue_mean(C, ns, R, S1)

    # Sigma^{t+1} = (1/N) sum_i [ n_i V_i + sum_j (m_ij - mu_hat)(m_ij - mu_hat)^T ]
    # with K_i = Sigma_t R_i^T C_i^{-1}, m_ij - mu_hat = K_i (y_ij - R_i mu_hat),
    # V_i = Sigma_t - K_i R_i Sigma_t.
    Sigma_next = jnp.zeros_like(Sigma_t)
    for i in range(len(R)):
        Ki = cho_solve(C[i], R[i] @ Sigma_t).T
        Vi = Sigma_t - Ki @ R[i] @ Sigma_t
        Sigma_next += ns[i] * Vi + Ki @ _scatter(i, mu_hat, ns, R, S1, S2) @ Ki.T
    Sigma_next = Sigma_next / jnp.sum(ns)
    return mu_hat, 0.5 * (Sigma_next + Sigma_next.T)  # symmetrize round-off


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@jit
def mlblue_solve(data, ns, R, Sigma, Gamma):
    """MLBLUE mean estimate given a known covariance Sigma (e.g. the true one)."""
    S1, _ = _sufficient_stats(data)
    return _mlblue_mean(_group_terms(Sigma, Gamma, R), ns, R, S1)


@jit
def em_mlblue(data, ns, R, Gamma, Sigma_guess, tol = 1e-13, max_iter = 1000):
    """Joint EM estimate of (mu, Sigma), compiled end to end.

    data        : list of K arrays, data[i] has shape (|G_i|, n_i)
    ns          : (K,) array of sample sizes n_i
    R           : list of K restriction matrices, R[i] has shape (|G_i|, L)
    Gamma       : (L, L) known noise covariance
    Sigma_guess : (L, L) SPD initial covariance

    Returns (mu, Sigma, n_iter, rel_change).
    """
    S1, S2 = _sufficient_stats(data)  # computed once; each iteration is O(K L^3), not O(N)

    def cond(state):
        it, _, rel_change = state
        return (it < max_iter) & (rel_change >= tol)

    def body(state):
        it, Sigma_t, _ = state
        _, Sigma_next = _em_step(Sigma_t, ns, R, Gamma, S1, S2)
        rel_change = jnp.linalg.norm(Sigma_next - Sigma_t) / jnp.linalg.norm(Sigma_t)
        return it + 1, Sigma_next, rel_change

    init = (jnp.array(0), Sigma_guess, jnp.array(jnp.inf, dtype = Sigma_guess.dtype))
    n_iter, Sigma_hat, rel_change = lax.while_loop(cond, body, init)

    # final mean at the converged covariance
    mu_hat = _mlblue_mean(_group_terms(Sigma_hat, Gamma, R), ns, R, S1)
    return mu_hat, Sigma_hat, n_iter, rel_change


@jit
def log_marginal_likelihood(data, ns, R, Gamma, mu, Sigma):
    """Observed-data log-likelihood; EM must never decrease this."""
    S1, S2 = _sufficient_stats(data)
    C = _group_terms(Sigma, Gamma, R)
    ll = 0.0
    for i in range(len(R)):
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(C[i][0])))
        quad = jnp.trace(cho_solve(C[i], _scatter(i, mu, ns, R, S1, S2)))
        ll += -0.5 * (ns[i] * (logdet + R[i].shape[0] * jnp.log(2 * pi)) + quad)
    return ll
