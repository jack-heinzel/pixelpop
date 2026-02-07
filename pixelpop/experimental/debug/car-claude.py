import numpy as np
from jax import lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as LSE
from jax.typing import ArrayLike

import numpyro
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.continuous import _is_sparse, _to_sparse
from numpyro.distributions.util import (
    promote_shapes,
    validate_sample,
)

from jax.scipy.special import gammaln
from functools import reduce
from ..models.car import initialize_ICAR, add_outer, mult_outer

from numpyro.distributions.transforms import (
    Transform,
    ComposeTransform,
    StickBreakingTransform, 
    ExpTransform, 
    AffineTransform
)

"""
Mixture probabilistic model for PixelPop.

Implements a mixture of a normalized nonparametric (PixelPop) component and
independent parametric marginals over the pixel dimensions:

    R(θ) = R₀ · [ξ · p_PP(θ_pixel) + (1 − ξ) · ∏ᵢ pᵢ(θ_pixel_i | λᵢ)] · ∏ⱼ pⱼ(θ_other_j | λⱼ)

where p_PP is a normalized ICAR field (σ-marginalized), ξ is a mixing
fraction, R₀ is the overall rate, and the "other" parametric models multiply
the entire mixture.

Usage
-----
This module provides:
    - ``ICAR_normalized``: σ-marginalized ICAR distribution constrained to
      integrate to 1 over the pixel grid (a proper probability density).
    - ``setup_mixture_probabilistic_model``: drop-in replacement for
      ``setup_probabilistic_model`` that builds the mixture model.

The user must supply a ``MixturePixelPopData`` dataclass (or a standard
``PixelPopData`` augmented with the extra fields) that specifies:
    - ``mixture_parameters``: parameters appearing in *both* the PixelPop grid
      and the parametric side of the mixture (e.g., ["mass_1", "mass_ratio"]).
    - ``mixture_parametric_models``: dict mapping each mixture parameter to a
      normalized log-density callable.
    - ``mixture_parameter_to_hyperparameters``: dict mapping each mixture
      parameter to a list of hyperparameter names.
    - ``common_strong_parameters``: parameters that multiply the whole mixture
      (e.g., ["chi_eff", "redshift"]).
"""

class VectorToGridTransform(Transform):
    """Reshape (..., n_tot) <-> (..., *event_shape)."""

    domain = constraints.real_vector
    codomain = constraints.real
    event_dim = 1

    def __init__(self, event_shape):
        super().__init__()
        self.event_shape = tuple(int(s) for s in event_shape)
        self.n_tot = int(np.prod(self.event_shape))

    def __call__(self, x):
        return jnp.reshape(x, x.shape[:-1] + self.event_shape)

    def _inverse(self, y):
        return jnp.reshape(y, y.shape[: -len(self.event_shape)] + (self.n_tot,))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.zeros(x.shape[:-1])

    def tree_flatten(self):
        return (), (self.event_shape,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (event_shape,) = aux_data
        return cls(event_shape)


class LogSimplexND(constraints.ParameterFreeConstraint):
    """
    N-dimensional log-simplex constraint.

    Enforces that the field sums (in log-space) to a target value,
    i.e., logsumexp(phi) == -log(dV), so that exp(phi) integrates to 1.
    """

    event_dim = 1

    def __init__(self, logsumexp: float, event_shape):
        self.logsumexp = logsumexp
        self.event_shape = tuple(int(s) for s in event_shape)
        self._ndim = len(self.event_shape)
        self._n_tot = int(np.prod(self.event_shape))

    def __call__(self, x):
        axes = tuple(range(-self._ndim, 0))
        x_sum = LSE(x, axis=axes)
        le0 = jnp.all(x <= 0, axis=axes)
        close = (x_sum < self.logsumexp + 1e-6) & (x_sum > self.logsumexp - 1e-6)
        return le0 & close

    def feasible_like(self, prototype):
        return jnp.full_like(prototype, self.logsumexp - jnp.log(self._n_tot))


# Register the biject_to transform for LogSimplexND
_biject_to = numpyro.distributions.transforms.biject_to


@_biject_to.register(LogSimplexND)
def _transform_to_logsimplex_nd(constraint):
    return ComposeTransform(
        [
            StickBreakingTransform(),
            ExpTransform().inv,
            AffineTransform(constraint.logsumexp, 1.0),
            VectorToGridTransform(constraint.event_shape),
        ]
    )


# ============================================================================
# Normalized, σ-marginalized ICAR distribution (Distribution-based)
# ============================================================================


def initialize_ICAR_normalized(dimension):
    """
    Factory that returns an ``ICAR_normalized`` distribution class for a
    given spatial dimension.

    The returned class is a NumPyro ``Distribution`` whose ``log_prob``
    evaluates the ICAR prior with:
    - an analytic marginalization over the global precision τ = 1/σ²
    - a normalization constraint enforcing ∫ exp(φ) dV = 1.

    For production use at high dimensions (e.g., 100×100), do NOT sample
    directly from this distribution (the ``StickBreaking`` transform is
    numerically fragile). Instead, use the ``log_prob`` method as a
    ``numpyro.factor`` on a manually normalized field — this is what
    ``setup_mixture_probabilistic_model`` does.

    For low-dimensional toy problems, ``numpyro.sample('x', icar_dist)``
    works fine.

    Parameters
    ----------
    dimension : int
        Number of spatial dimensions of the pixel grid.

    Returns
    -------
    ICAR_normalized : class
        NumPyro ``Distribution`` subclass.
    """

    class ICAR_normalized(Distribution):
        """
        σ-marginalized, normalized ICAR distribution.

        The field φ lives on a log-simplex such that
        ``logsumexp(φ) = -log(dV)``, ensuring ``∑ exp(φ) · dV = 1``.
        """

        arg_constraints = {
            "single_dimension_adj_matrices": constraints.independent(
                constraints.dependent(is_discrete=False, event_dim=2), 1
            ),
            "dx": constraints.positive,
        }
        reparametrized_params = ["single_dimension_adj_matrices", "dx"]
        pytree_aux_fields = (
            "is_sparse",
            "single_dimension_adj_matrices",
            "dx",
        )

        def __init__(
            self,
            single_dimension_adj_matrices,
            *,
            is_sparse=False,
            dx=1.0,
            validate_args=None,
        ):
            self.is_sparse = is_sparse
            self.dx = dx
            batch_shape = ()

            self.single_dimension_adj_matrices = []
            if self.is_sparse:
                for mat in single_dimension_adj_matrices:
                    if mat.ndim != 2:
                        raise ValueError(
                            "Only 2-dimensional adjacency matrices are "
                            "supported."
                        )
                    if not (
                        isinstance(mat, np.ndarray) or _is_sparse(mat)
                    ):
                        raise ValueError(
                            "Adjacency matrix must be a numpy array or "
                            "scipy sparse matrix."
                        )
                    self.single_dimension_adj_matrices.append(_to_sparse(mat))
            else:
                for mat in single_dimension_adj_matrices:
                    assert not _is_sparse(mat), (
                        "Adjacency matrix is sparse — set is_sparse=True."
                    )
                    (mat,) = promote_shapes(
                        mat, shape=batch_shape + mat.shape[-2:]
                    )
                    self.single_dimension_adj_matrices.append(mat)

            event_shape = tuple(
                jnp.shape(m)[-1] for m in self.single_dimension_adj_matrices
            )
            super().__init__(
                batch_shape=batch_shape,
                event_shape=event_shape,
                validate_args=validate_args,
            )

        @property
        def support(self):
            return LogSimplexND(
                logsumexp=-jnp.log(self.dx),
                event_shape=self.event_shape,
            )

        def sample(self, key, sample_shape=()):
            raise NotImplementedError(
                "Cannot sample from normalized ICAR "
                "(constrained distribution)."
            )

        @validate_sample
        def log_prob(self, phi):
            lams = []
            prec_mat = []
            n = 1

            for ii, mat in enumerate(self.single_dimension_adj_matrices):
                if self.is_sparse:
                    D = np.asarray(mat.sum(axis=-1)).squeeze(axis=-1)
                    Q_single = np.diag(D) - mat.toarray()
                else:
                    D = mat.sum(axis=-1)
                    Q_single = jnp.diag(D) - mat

                n *= D.shape[-1]

                if isinstance(Q_single, np.ndarray):
                    lam = np.linalg.eigvalsh(Q_single)
                    lam[0] = 0.0
                else:
                    lam = jnp.linalg.eigvalsh(Q_single)
                    lam = lam.at[0].set(0.0)

                prec_mat.append(jnp.asarray(Q_single))
                lams.append(lam)

            lams = [jnp.asarray(l) for l in lams]
            ar = reduce(add_outer, lams)
            logdet = jnp.sum(
                jnp.log(ar.at[(0,) * dimension].set(1.0))
            )

            logquad = 0.0
            for ii in range(dimension):
                z = jnp.moveaxis(
                    jnp.tensordot(
                        prec_mat[ii],
                        jnp.moveaxis(phi, ii, 0),
                        axes=(0, 0),
                    ),
                    0,
                    ii,
                )
                logquad += jnp.tensordot(z, phi, axes=dimension)

            r = n - 1
            log_marg = (
                0.5 * r * (jnp.log(2) - jnp.log(logquad))
                + gammaln(r / 2)
                - jnp.log(2)
            )
            return 0.5 * (-r * jnp.log(2 * jnp.pi) + logdet) + log_marg

        @staticmethod
        def infer_shapes(single_dimension_adj_matrices):
            event_shape = tuple(
                jnp.shape(m)[-1] for m in single_dimension_adj_matrices
            )
            batch_shape = lax.broadcast_shapes(
                *[jnp.shape(m)[:-2] for m in single_dimension_adj_matrices]
            )
            return batch_shape, event_shape

    return ICAR_normalized



class DiagonalizedICARTransform:
    '''
    TODO: can we use the numpyro syntax more natively for this transform?

    Relies on structure of kronecker sum to rapidly compute eigenbasis for the 
    full adjacency structure, then sample in the diagonalized eigenbasis
    '''
    def __init__(
            self, 
            log_sigmas,
            single_dimension_adj_matrices, 
            is_sparse=False
            ):


        precision_mats = []
        eigenvalue_list = []
        self.eigenvector_list = []
        self.dimension = len(single_dimension_adj_matrices)
        if jnp.ndim(log_sigmas) == 0:
            (log_sigmas,) = promote_shapes(log_sigmas, shape=(self.dimension,))
        
        self.log_sigmas = log_sigmas
        precs = jnp.exp(-2*self.log_sigmas)
        
        for ii, single_dimension_adj_matrix in enumerate(single_dimension_adj_matrices):
            if is_sparse:
                precision_mat = jnp.array(single_dimension_adj_matrix.toarray())
            else:
                assert not _is_sparse(single_dimension_adj_matrix), (
                    "single_dimension_adj_matrix is a sparse matrix so please specify `is_sparse=True`."
                )
        
            D = jnp.diag(jnp.sum(precision_mat, axis=1))
            precision_mat = D - precision_mat
            
            eig_result = jnp.linalg.eigh(precision_mat, )
            eigenvalues = eig_result.eigenvalues
            eigenvalues = eigenvalues.at[0].set(0.)
            eigenvectors = eig_result.eigenvectors
            
            precision_mats.append(precision_mat)
            eigenvalue_list.append(precs[ii]*eigenvalues)
            self.eigenvector_list.append(eigenvectors)

        self.multiD_eigenvalues = reduce(add_outer, eigenvalue_list)
        self.multiD_eigenvalues = self.multiD_eigenvalues.at[(0,)*self.dimension].set(jnp.sum(precs))
        # to fix the scale, otherwise divide by zero bc improper

    def __call__(self, eigenbasis):
        '''
        slick way to calculate the sum we want:

        \sum_{ijk...} \alpha[i,j,k,...] v_1[i] \otimes v_2[j] \otimes v_3[k] \otimes ...

        where v_1, v_2, ... are the eigenvectors 
        '''

        res = eigenbasis * self.multiD_eigenvalues ** (-1/2) 
        for v in self.eigenvector_list:
            res = jnp.tensordot(res, v, axes=(0, 1))

        return res

    def log_prob(self, eigenbasis):
        '''
        log prob in transformed Cartesian basis.
        '''
        if isinstance(eigenbasis, jnp.ndarray):
            eigenbasis = eigenbasis.at[(0,)*jnp.ndim(eigenbasis)].set(0.)
        elif isinstance(eigenbasis, np.ndarray):
            eigenbasis[(0,)*jnp.ndim(eigenbasis)] = 0.
        std_lp = -jnp.sum(eigenbasis**2) / 2 - np.log(2*np.pi) * eigenbasis.size / 2
        return std_lp + 0.5*jnp.sum(jnp.log(self.multiD_eigenvalues))


def lower_triangular_sigma_marg_log_prob_and_log_quad(phi, n, single_dimension_adj_matrices):
    """
    Compute the log-probability for an ICAR prior with a lower-triangular parameterization.

    This function evaluates the quadratic form and normalization term of the
    ICAR log-density given adjacency matrices and a scalar log-scale.

    Parameters
    ----------
    phi : jnp.ndarray
        Field values on the spatial grid.
    n : int
        Total number of sites, e.g., the number of degrees of freedom, n = bins * (bins+1) / 2).
    single_dimension_adj_matrices : list of ndarray or sparse matrices
        List of adjacency matrices, one for each spatial dimension.

    Returns
    -------
    tuple of floats
        The log-probability of `phi` under the ICAR prior, and logquad for producing conditional sigma 
        samples
    """

    dimension = len(single_dimension_adj_matrices)
    prec_mat = []
    for ii, single_dimension_adj_matrix in enumerate(single_dimension_adj_matrices):
        D = np.asarray(single_dimension_adj_matrix.sum(axis=-1))
        scaled_single_prec = np.diag(D) - single_dimension_adj_matrix.toarray()
        prec_mat.append(jnp.asarray(scaled_single_prec))

    logquad = 0.
    for ii in range(dimension):
        z = jnp.moveaxis(
            jnp.tensordot(prec_mat[ii], jnp.moveaxis(phi,ii,0), axes=(0,0)), # tensordot requires a concrete axis ... can I get this in a jitted fn?
            0,ii)
        step = jnp.tensordot(z, phi, axes=dimension)
        logquad += step
    

    log_marg_term = 0.5 * n * (jnp.log(2) - jnp.log(logquad)) + gammaln(n / 2) - jnp.log(2)

    return -0.5 * n * jnp.log(2*jnp.pi) + log_marg_term, logquad


def lower_triangular_sigma_marg_log_prob(phi, n, single_dimension_adj_matrices):
    """
    Compute the log-probability for an ICAR prior with a lower-triangular parameterization.

    This function evaluates the quadratic form and normalization term of the
    ICAR log-density given adjacency matrices and a scalar log-scale.

    Parameters
    ----------
    phi : jnp.ndarray
        Field values on the spatial grid.
    n : int
        Total number of sites, e.g., the number of degrees of freedom, n = bins * (bins+1) / 2).
    log_sigma : float
        Log standard deviation of the prior.
    single_dimension_adj_matrices : list of ndarray or sparse matrices
        List of adjacency matrices, one for each spatial dimension.

    Returns
    -------
    float
        The log-probability of `phi` under the ICAR prior.
    """

    lp, lq = lower_triangular_sigma_marg_log_prob_and_log_quad(phi, n, single_dimension_adj_matrices)
    return lp