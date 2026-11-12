"""Frozen task-level MPPCA source from TOPS, Eq. (4).

EM alternates mixture responsibilities and the weighted PPCA maximum-likelihood
solution: residual variance is the mean discarded eigenvalue, and retained
loading eigenvalues are max(eigenvalue - residual variance, 0).
"""
import distrax
from flax import struct
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


@struct.dataclass
class MPPCA:
    weights: object
    means: object
    loadings: object
    residual_variances: object
    scale_tril: object

    def distribution(self, batch_shape=()):
        # A cached Cholesky factor samples exactly N(mu, WW^T + sigma^2 I).
        # Broadcast components over observations: sources are task-conditioned,
        # independent of the observation and of the paired demonstration.
        k, d = self.means.shape
        components = distrax.MultivariateNormalTri(
            loc=jnp.broadcast_to(self.means, (*batch_shape, k, d)),
            scale_tri=jnp.broadcast_to(self.scale_tril, (*batch_shape, k, d, d)))
        return distrax.MixtureSameFamily(
            distrax.Categorical(probs=jnp.broadcast_to(self.weights, (*batch_shape, k))),
            components)


def fit_mppca(samples, *, components=2, rank=20, max_iter=100, tol=1e-4,
              seed=0, source_std=1.0, variance_floor=1e-6):
    """Fit by EM, then calibrate pooled std to the warm-up source (TOPS A.5).

    Scale all means and component deviations about the pooled mean; this
    preserves the fitted center and makes sqrt(trace(mixture covariance)/D)
    equal source_std. The OGPO warm-up source is N(0, I), so its scale is 1.
    """
    for name, value, minimum in (('components', components, 1),
                                 ('rank', rank, 0), ('max_iter', max_iter, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if any(not np.isfinite(v) or v <= 0 for v in (tol, source_std, variance_floor)):
        raise ValueError('tol, source_std and variance_floor must be finite and positive')
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1 or len(x) < components or not np.isfinite(x).all():
        raise ValueError('MPPCA needs finite [N, D] samples with N >= components')
    n, d = x.shape
    rank = min(rank, d - 1)
    labels = KMeans(n_clusters=components, n_init=3, random_state=seed).fit_predict(x)
    resp = np.eye(components)[labels]
    history = []
    for iteration in range(max_iter + 1):
        counts = resp.sum(0)
        weights = np.maximum(counts, 1e-12)
        weights /= weights.sum()
        means, loadings, residuals, factors, log_joint = [], [], [], [], []
        for k in range(components):
            # A collapsed component is reset to a deterministic data point and
            # the global covariance, retaining a positive (tiny) mixture weight.
            if counts[k] <= 1e-10:
                mean = x[(seed + k) % n]
                centered = x - x.mean(0)
                covariance = centered.T @ centered / n
            else:
                mean = resp[:, k] @ x / counts[k]
                centered = x - mean
                covariance = (centered.T * resp[:, k]) @ centered / counts[k]
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            residual = max(float(eigenvalues[:d - rank].mean()), variance_floor)
            loading = (eigenvectors[:, d - rank:] *
                       np.sqrt(np.maximum(eigenvalues[d - rank:] - residual, 0)))
            covariance = loading @ loading.T + residual * np.eye(d)
            factor = np.linalg.cholesky(covariance)
            diff = x - mean
            # Woodbury inverse, avoiding a triangular solve for every sample.
            projection = diff @ loading
            quadratic = (np.square(diff).sum(-1) -
                         (np.square(projection) /
                          (residual + np.square(loading).sum(0))).sum(-1)) / residual
            logdet = 2 * np.log(np.diag(factor)).sum()
            log_joint.append(np.log(weights[k]) - .5 * (d * np.log(2 * np.pi) + logdet + quadratic))
            means.append(mean)
            loadings.append(loading)
            residuals.append(residual)
            factors.append(factor)
        log_joint = np.stack(log_joint, axis=1)
        normalizer = logsumexp(log_joint, axis=1)
        history.append(float(normalizer.mean()))
        if not np.isfinite(history[-1]):
            raise ValueError('Nonfinite MPPCA log likelihood')
        if iteration == max_iter or (len(history) > 1 and
                abs(history[-1] - history[-2]) <= tol):
            break
        resp = np.exp(log_joint - normalizer[:, None])
    means, loadings, residuals, factors = map(np.asarray, (means, loadings, residuals, factors))
    center = weights @ means
    pooled_variance = np.sum(weights * (np.square(loadings).sum((1, 2)) +
                              d * residuals + np.square(means - center).sum(-1))) / d
    scale = source_std / np.sqrt(pooled_variance)
    means = center + scale * (means - center)
    model = MPPCA(*[jnp.asarray(a, dtype=jnp.float32) for a in (
        weights, means, loadings * scale, residuals * scale**2, factors * scale)])
    return model, dict(mppca_log_likelihood=history[-1], mppca_em_iterations=iteration,
                       mppca_components=components, mppca_rank=rank,
                       mppca_source_std=source_std)
