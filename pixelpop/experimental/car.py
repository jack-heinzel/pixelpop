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
    ComposeTransform,
    StickBreakingTransform, 
    ExpTransform, 
    AffineTransform
)

def initialize_sigma_marginalized_ICAR(dimension):
    """
    Construct an Intrinsic Conditional Autoregressive (ICAR) distribution class.
    Here, the log-sigma is analytically marginalized over.

    The returned class defines a NumPyro-compatible ICAR prior. 
    The ICAR is a special case of a Gaussian Markov random field where the 
    precision matrix is determined by adjacency matrices of spatial sites.

    For a single log-sigma parameter, the integral over the improper prior
    \pi(\sigma) \propto 1/\sigma gives

    \int \frac{1}{\sigma}\frac{1}{\sigma^{n}} \exp(-\frac{x}{2\sigma^2}) d\sigma 
    = 
    2^{n/2 - 1} x^{-n/2} \Gamma(n/2)
    
    Parameters
    ----------
    dimension : int
        Number of spatial dimensions.

    Returns
    -------
    ICAR_sigma_marg : class
        A NumPyro `Distribution` subclass implementing the ICAR prior, with
        methods for `log_prob`, shape inference, and JAX pytree compatibility.

    Notes
    -----
    - The distribution is improper (cannot sample directly).
    - Adjacency matrices must be symmetric with all sites having neighbors.
    - Sparse and dense adjacency matrix representations are supported.
    """
    base_ICAR_class = initialize_ICAR(dimension, length_scales=False)

    class ICAR_marginalized_sigma(base_ICAR_class):

        def __init__(
            self, 
            single_dimension_adj_matrices,
            *args,
            is_sparse=False,
            validate_args=None,
        ):
            base_lsigma = 0.
            super(ICAR_marginalized_sigma, self).__init__(
                base_lsigma,
                single_dimension_adj_matrices,
                *args,
                is_sparse=is_sparse,
                validate_args=validate_args,
            )
        
        @validate_sample
        def log_prob(self, phi):

            lams = []
            prec_mat = []
            n = 1
            for ii, single_dimension_adj_matrix in enumerate(self.single_dimension_adj_matrices):
                if self.is_sparse:
                    D = np.asarray(single_dimension_adj_matrix.sum(axis=-1)).squeeze(axis=-1)
                    scaled_single_prec = np.diag(D) - single_dimension_adj_matrix.toarray()
                else:
                    D = single_dimension_adj_matrix.sum(axis=-1)# .squeeze(axis=0)
                    scaled_single_prec = jnp.diag(D) - single_dimension_adj_matrix
                
                n *= D.shape[-1]
                # TODO: look into sparse eigenvalue methods
                if isinstance(scaled_single_prec, np.ndarray):
                    lam = np.linalg.eigvalsh(scaled_single_prec)   
                    lam[0] = 0. # set to zero, otherwise float precision can allow this to be negative and cause problems
                    
                else:
                    print(jnp.diag(D).shape, single_dimension_adj_matrix.shape)
                    lam = jnp.linalg.eigvalsh(scaled_single_prec)
                    lam = lam.at[0].set(0.) # set to zero, otherwise float precision can allow this to be negative and cause problems
                prec_mat.append(jnp.asarray(scaled_single_prec))
                lams.append(lam)

            ar = reduce(add_outer, lams)
            
            logdet = jnp.sum(jnp.log(ar.at[(0,)*dimension].set(1.)))
            logquad = 0.
            for ii in range(dimension):
                z = jnp.moveaxis(
                    jnp.tensordot(prec_mat[ii], jnp.moveaxis(phi,ii,0), axes=(0,0)), # tensordot requires a concrete axis ... can I get this in a jitted fn?
                    0,ii)
                
                step = jnp.tensordot(z, phi, axes=dimension)
                logquad += step

            log_marg_term = 0.5 * n * (jnp.log(2) - jnp.log(logquad)) + gammaln(n / 2) - jnp.log(2)

            return 0.5 * (-n * jnp.log(2*jnp.pi) + logdet) + log_marg_term
        
        @staticmethod
        def infer_shapes(single_dimension_adj_matrices):
            event_shape = tuple([jnp.shape(mat)[-1] for mat in single_dimension_adj_matrices])
            batch_shape = lax.broadcast_shapes(
                *[jnp.shape(mat)[:-2] for mat in single_dimension_adj_matrices]
            )
            return batch_shape, event_shape

    return ICAR_marginalized_sigma
    

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


class _LogSimplex(constraints.ParameterFreeConstraint):
    event_dim = 1
    def __init__(self, logsumexp: ArrayLike) -> None:
        self.logsumexp = logsumexp

    def __call__(self, x: ArrayLike) -> ArrayLike:
        x_sum = LSE(x, axis=-1)
        return (x <= 0).all(axis=-1) & (x_sum < self.logsumexp + 1e-6) & (x_sum > self.logsumexp - 1e-6)

    def feasible_like(self, prototype: ArrayLike) -> ArrayLike:
        return jnp.full_like(prototype, self.logsumexp - jnp.log(prototype.shape[-1]))

logsimplex = _LogSimplex
biject_to = numpyro.distributions.transforms.biject_to

@biject_to.register(logsimplex)
def _transform_to_logsimplex(constraint):
    return ComposeTransform([
        StickBreakingTransform(), 
        ExpTransform().inv, 
        AffineTransform(constraint.logsumexp, 1.)
        ])


class ICAR_normalized(Distribution):
    '''
    
    TODO IMPLEMENT IN TERMS OF ONE D ADJ MATRICES

    '''
    arg_constraints = {
        "log_sigma": constraints.real,
        "adj_matrix": constraints.dependent(is_discrete=False, event_dim=2),
        "dx": constraints.positive,
    }
    reparametrized_params = [
        "log_sigma",
        "adj_matrix",
        "dx",
    ]

    pytree_aux_fields = ("is_sparse", "adj_matrix", "dx")

    def __init__(
        self,
        log_sigma,
        adj_matrix,
        *,
        is_sparse=False,
        dx=1., 
        validate_args=None,
    ):
        
        assert jnp.ndim(log_sigma) == 0
        self.is_sparse = is_sparse
        self.dx = dx
        batch_shape = ()
        # print('batch shape is ', batch_shape)
        if self.is_sparse:
            if adj_matrix.ndim != 2:
                raise ValueError(
                    "Currently, we only support 2-dimensional adj_matrix. Please make a feature request",
                    " if you need higher dimensional adj_matrix.",
                )
            if not (isinstance(adj_matrix, np.ndarray) or _is_sparse(adj_matrix)):
                raise ValueError(
                    "adj_matrix needs to be a numpy array or a scipy sparse matrix. Please make a feature",
                    " request if you need to support jax ndarrays.",
                )
            self.adj_matrix = adj_matrix
            # TODO: look into future jax sparse csr functionality and other developments
        else:
            assert not _is_sparse(adj_matrix), (
                "single_dimension_adj_matrix is a sparse matrix so please specify `is_sparse=True`."
            )
                # TODO: look into static jax ndarray representation
            (adj_matrix,) = promote_shapes(
                adj_matrix, shape=batch_shape + adj_matrix.shape[-2:]
            )
            self.adj_matrix = adj_matrix

        event_shape = (jnp.shape(adj_matrix)[-1],)

        (self.log_sigma,) = promote_shapes(log_sigma, shape=batch_shape)

        super(ICAR_normalized, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

        if self._validate_args and (isinstance(self.adj_matrix, np.ndarray) or is_sparse):
            assert (self.adj_matrix.sum(axis=-1) > 0).all() > 0, (
                "all sites in adjacency matrix must have neighbours"
            )

            if self.is_sparse:
                assert (self.adj_matrix != self.adj_matrix.T).nnz == 0, (
                    "adjacency matrix must be symmetric"
                )
            else:
                assert np.array_equal(
                    self.adj_matrix, np.swapaxes(self.adj_matrix, -2, -1)
                ), "adjacency matrix must be symmetric"

    @property
    def support(self) -> constraints.Constraint:
        return logsimplex(-jnp.log(self.dx))
    
    def sample(self, key, sample_shape=()):
        # cannot sample from an improper distribution
        raise NotImplementedError 

    @validate_sample
    def log_prob(self, phi):

        adj_matrix = self.adj_matrix
        conditional_precision = jnp.exp(-2*self.log_sigma)
        if self.is_sparse:
            D = np.asarray(adj_matrix.sum(axis=-1))
            adj_matrix = BCOO.from_scipy_sparse(adj_matrix)
        else:
            D = adj_matrix.sum(axis=-1)

        n = D.shape[-1]

        logprec = -2 * (n-1) * self.log_sigma

        logquad = conditional_precision * jnp.sum(
            phi
            * (
                D * phi
                - (adj_matrix @ phi[..., jnp.newaxis]).squeeze(axis=-1)
            ),
            -1,
        )

        return 0.5 * (logprec - logquad)
    
    @staticmethod
    def infer_shapes(log_sigma, adj_matrix):
        event_shape = jnp.shape(adj_matrix)[-1]
        batch_shape = lax.broadcast_shapes(
            jnp.shape(log_sigma)[:-1], jnp.shape(adj_matrix)[:-2]
        )
        return batch_shape, event_shape

    def tree_flatten(self):
        data, aux = super().tree_flatten()
        single_dimension_adj_matrix_data_idx = type(self).gather_pytree_data_fields().index("adj_matrix")
        single_dimension_adj_matrix_aux_idx = type(self).gather_pytree_aux_fields().index("adj_matrix")

        if not self.is_sparse:
            aux = list(aux)
            aux[single_dimension_adj_matrix_aux_idx] = None
            aux = tuple(aux)
        else:
            data = list(data)
            data[single_dimension_adj_matrix_data_idx] = None
            data = tuple(data)
        return data, aux

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        d = super().tree_unflatten(aux_data, params)
        if not d.is_sparse:
            adj_matrix_data_idx = cls.gather_pytree_data_fields().index("adj_matrix")
            setattr(d, "adj_matrix", params[adj_matrix_data_idx])
        else:
            adj_matrix_aux_idx = cls.gather_pytree_aux_fields().index("adj_matrix")
            setattr(d, "adj_matrix", aux_data[adj_matrix_aux_idx])
        return d

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

    return -0.5 * n * jnp.log(2*jnp.pi) + log_marg_term
