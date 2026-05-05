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

        def log_prob_and_quad(self, phi):
            # same as log_prob, but also returns logquad for producing sigma (coupling) samples from conditional Gamma distribution

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

            return 0.5 * (-n * jnp.log(2*jnp.pi) + logdet) + log_marg_term, logquad
        
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

class VectorToGridTransform(Transform):
    """
    Reshape (..., n_tot) <-> (..., *event_shape)
    """
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
        return jnp.reshape(y, y.shape[:-len(self.event_shape)] + (self.n_tot,))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.zeros(x.shape[:-1])

    def tree_flatten(self):
        return (), (self.event_shape,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (event_shape,) = aux_data
        return cls(event_shape)
        

class _LogSimplexNDim(constraints.ParameterFreeConstraint):
    """
    N-dimensional generalization of _LogSimplex implemented by Jack
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

        #le0 = jnp.all(x <= 0, axis=axes)

        close = (x_sum < self.logsumexp + 1e-6) & (x_sum > self.logsumexp - 1e-6)
        #return le0 & close
        return close

    def feasible_like(self, prototype):
        return jnp.full_like(prototype, self.logsumexp - jnp.log(self._n_tot))

logsimplexNDim = _LogSimplexNDim
biject_to = numpyro.distributions.transforms.biject_to

@biject_to.register(_LogSimplexNDim)
def _transform_to_logsimplex_nd(constraint):
    return ComposeTransform([
        StickBreakingTransform(),                    
        ExpTransform().inv,                          
        AffineTransform(constraint.logsumexp, 1.), 
        VectorToGridTransform(constraint.event_shape)
    ])

def initialize_ICAR_normalized(dimension):
    
    base_ICAR_class = initialize_ICAR(dimension, length_scales=False)

    class ICAR_normalized(base_ICAR_class):
        """
        Implements a normalized version of the ICAR model, marginalized over the sigma coupling parameter.
        Sofia note: in Ndim, dx becomes dx*dy*...*dN, so maybe dV is a better name for the parameter?
        """

        arg_constraints = {
            "single_dimension_adj_matrices": constraints.independent(
                constraints.dependent(is_discrete=False, event_dim=2), 1
            ),
            "dx": constraints.positive,
        }
        # arg_constraints = {
        #     "single_dimension_adj_matrices": constraints.dependent(is_discrete=False, event_dim=2),
        #     "dx": constraints.positive,
        # }
        reparametrized_params = ["single_dimension_adj_matrices", "dx"]
        pytree_aux_fields = ("is_sparse", "single_dimension_adj_matrices", "dx")

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

            # store adjacency matrices (sparse or dense)
            self.single_dimension_adj_matrices = []
            if self.is_sparse:
                for single_dimension_adj_matrix in single_dimension_adj_matrices:
                    if single_dimension_adj_matrix.ndim != 2:
                        raise ValueError(
                            "Currently, we only support 2-dimensional adj_matrix. Please make a feature request"
                            " if you need higher dimensional adj_matrix."
                        )
                    if not (
                        isinstance(single_dimension_adj_matrix, np.ndarray)
                        or _is_sparse(single_dimension_adj_matrix)
                    ):
                        raise ValueError(
                            "adj_matrix needs to be a numpy array or a scipy sparse matrix. Please make a feature"
                            " request if you need to support jax ndarrays."
                        )
                    # TODO: look into future jax sparse csr functionality and other developments
                    self.single_dimension_adj_matrices.append(
                        _to_sparse(single_dimension_adj_matrix)
                    )
            else:
                for single_dimension_adj_matrix in single_dimension_adj_matrices:
                    assert not _is_sparse(single_dimension_adj_matrix), (
                        "single_dimension_adj_matrix is a sparse matrix so please specify `is_sparse=True`."
                    )
                    # TODO: look into static jax ndarray representation
                    (single_dimension_adj_matrix,) = promote_shapes(
                        single_dimension_adj_matrix,
                        shape=batch_shape + single_dimension_adj_matrix.shape[-2:],
                    )
                    self.single_dimension_adj_matrices.append(single_dimension_adj_matrix)

            event_shape = tuple(
                [jnp.shape(mat)[-1] for mat in self.single_dimension_adj_matrices]
            )

            # super(ICAR_normalized, self).__init__(
            #     batch_shape=batch_shape,
            #     event_shape=event_shape,
            #     validate_args=validate_args,
            # )
            Distribution.__init__(
                self,
                batch_shape=batch_shape,
                event_shape=event_shape,
                validate_args=validate_args,
            )

        @property
        def support(self) -> constraints.Constraint:
            return _LogSimplexNDim(
                logsumexp=-jnp.log(self.dx), event_shape=self.event_shape
            )

        def sample(self, key, sample_shape=()):
            # cannot sample from an improper distribution
            raise NotImplementedError

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
                    D = single_dimension_adj_matrix.sum(axis=-1)
                    scaled_single_prec = jnp.diag(D) - single_dimension_adj_matrix

                n *= D.shape[-1]
                if isinstance(scaled_single_prec, np.ndarray):
                    lam = np.linalg.eigvalsh(scaled_single_prec)
                    # set to zero, otherwise float precision can allow this to be negative and cause problems
                    lam[0] = 0.0
                else:
                    # print(jnp.diag(D).shape, single_dimension_adj_matrix.shape)
                    lam = jnp.linalg.eigvalsh(scaled_single_prec)
                    lam = lam.at[0].set(0.0)

                prec_mat.append(jnp.asarray(scaled_single_prec))
                lams.append(lam)

            lams = jnp.array(lams)
            ar = reduce(add_outer, lams)
            logdet = jnp.sum(jnp.log(ar.at[(0,) * dimension].set(1.0)))

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
                step = jnp.tensordot(z, phi, axes=dimension)
                logquad += step

            log_marg_term = (
                0.5 * (n - 1) * (jnp.log(2) - jnp.log(logquad))
                + gammaln((n - 1) / 2)
                - jnp.log(2)
            )
            return 0.5 * (-(n - 1) * jnp.log(2 * jnp.pi) + logdet) + log_marg_term

        
        def log_prob_and_quad(self, phi):
            # same as log_prob, but also returns logquad for producing sigma (coupling) samples from conditional Gamma distribution
            lams = []
            prec_mat = []
            n = 1
            for ii, single_dimension_adj_matrix in enumerate(self.single_dimension_adj_matrices):
                if self.is_sparse:
                    D = np.asarray(single_dimension_adj_matrix.sum(axis=-1)).squeeze(axis=-1)
                    scaled_single_prec = np.diag(D) - single_dimension_adj_matrix.toarray()
                else:
                    D = single_dimension_adj_matrix.sum(axis=-1)
                    scaled_single_prec = jnp.diag(D) - single_dimension_adj_matrix

                n *= D.shape[-1]
                if isinstance(scaled_single_prec, np.ndarray):
                    lam = np.linalg.eigvalsh(scaled_single_prec)
                    # set to zero, otherwise float precision can allow this to be negative and cause problems
                    lam[0] = 0.0
                else:
                    # print(jnp.diag(D).shape, single_dimension_adj_matrix.shape)
                    lam = jnp.linalg.eigvalsh(scaled_single_prec)
                    lam = lam.at[0].set(0.0)

                prec_mat.append(jnp.asarray(scaled_single_prec))
                lams.append(lam)

            lams = jnp.array(lams)
            ar = reduce(add_outer, lams)
            logdet = jnp.sum(jnp.log(ar.at[(0,) * dimension].set(1.0)))

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
                step = jnp.tensordot(z, phi, axes=dimension)
                logquad += step

            log_marg_term = (
                0.5 * (n - 1) * (jnp.log(2) - jnp.log(logquad))
                + gammaln((n - 1) / 2)
                - jnp.log(2)
            )
            return 0.5 * (-(n - 1) * jnp.log(2 * jnp.pi) + logdet) + log_marg_term, logquad

        def quad_form(self, phi):
            """
            Compute only the quadratic form φᵀQφ (no eigenvalues, no logdet).

            Use this to recover σ samples via the conditional Gamma distribution
            without recomputing the full log_prob (which numpyro.sample already
            evaluates).

            Parameters
            ----------
            phi : jnp.ndarray
                Field values on the spatial grid.

            Returns
            -------
            logquad : float
                The quadratic form φᵀQφ summed over all dimensions.
            """
            prec_mat = []
            for ii, single_dimension_adj_matrix in enumerate(self.single_dimension_adj_matrices):
                if self.is_sparse:
                    D = np.asarray(single_dimension_adj_matrix.sum(axis=-1)).squeeze(axis=-1)
                    scaled_single_prec = np.diag(D) - single_dimension_adj_matrix.toarray()
                else:
                    D = single_dimension_adj_matrix.sum(axis=-1)
                    scaled_single_prec = jnp.diag(D) - single_dimension_adj_matrix
                prec_mat.append(jnp.asarray(scaled_single_prec))

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
                step = jnp.tensordot(z, phi, axes=dimension)
                logquad += step

            return logquad

        @staticmethod
        def infer_shapes(single_dimension_adj_matrices):
            event_shape = tuple([jnp.shape(mat)[-1] for mat in single_dimension_adj_matrices])
            batch_shape = lax.broadcast_shapes(
                *[jnp.shape(mat)[:-2] for mat in single_dimension_adj_matrices]
            )
            return batch_shape, event_shape

    return ICAR_normalized


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


def _lt_normalized_log_prob_and_log_quad(
    phi_full, single_dimension_adj_matrices, dimension, n_dof
):
    """
    Sigma-marginalized ICAR log-prior for the LT + LSE-normalized PixelPop case.

    Mirrors `initialize_ICAR_normalized.log_prob` (the working non-LT normalized
    branch): includes the eigenvalue-based log-det of the full-grid precision
    and uses (n_dof) directly as the effective DOF (already accounting for the
    LSE constraint -- pass `tri_size - 1` for the LT case).

    Parameters
    ----------
    phi_full : jnp.ndarray
        Full mirrored grid (output of lt_map applied to the LT vector).
        Symmetric in the first two axes, so phi_full[i,j] = phi_full[j,i].
    single_dimension_adj_matrices : list of ndarray or sparse matrices
        Adjacency matrices, one per spatial dimension.
    dimension : int
        Number of spatial dimensions.
    n_dof : int
        Effective DOF of the LT-constrained Gaussian (tri_size - 1 for 2D LT).
    """
    lams = []
    prec_mat = []
    for ii, single_dimension_adj_matrix in enumerate(single_dimension_adj_matrices):
        D = np.asarray(single_dimension_adj_matrix.sum(axis=-1)).squeeze(axis=-1)
        scaled_single_prec = np.diag(D) - single_dimension_adj_matrix.toarray()

        lam = np.linalg.eigvalsh(scaled_single_prec)
        lam[0] = 0.0

        prec_mat.append(jnp.asarray(scaled_single_prec))
        lams.append(lam)

    lams = jnp.array(lams)
    ar = reduce(add_outer, lams)
    logdet = jnp.sum(jnp.log(ar.at[(0,) * dimension].set(1.0)))

    logquad = 0.0
    for ii in range(dimension):
        z = jnp.moveaxis(
            jnp.tensordot(prec_mat[ii], jnp.moveaxis(phi_full, ii, 0), axes=(0, 0)),
            0,
            ii,
        )
        step = jnp.tensordot(z, phi_full, axes=dimension)
        logquad += step

    log_marg_term = (
        0.5 * n_dof * (jnp.log(2) - jnp.log(logquad))
        + gammaln(n_dof / 2)
        - jnp.log(2)
    )
    log_prob = 0.5 * (-n_dof * jnp.log(2 * jnp.pi) + logdet) + log_marg_term
    return log_prob, logquad


def initialize_ICAR_normalized_lower_triangular(dimension, bins_first):
    """
    Construct a normalized + sigma-marginalized + lower-triangular ICAR distribution.

    The sample lives on the unique lower-triangular vector of the first two
    dimensions (length `tri_size = bins_first * (bins_first + 1) / 2`, optionally
    extended by `bins[2:]`), constrained to be a normalized log-density over the
    LT region: LSE(phi_LT) + log(dx) = 0.

    Internally the class applies `lower_triangular_map(bins_first)` to convert
    the LT vector to the full mirrored bins-by-bins grid before evaluating the
    sigma-marginalized prior factor (mirroring `initialize_ICAR_normalized.log_prob`).

    Parameters
    ----------
    dimension : int
        Total number of spatial dimensions of the underlying ICAR field.
    bins_first : int
        Number of bins in the first (and second) dimensions; required because
        `lower_triangular_map` is built statically for that size.

    Returns
    -------
    ICAR_normalized_LT : class
        NumPyro `Distribution` subclass. Sampling produces an LT vector; the
        upper-triangular indices are obtained by applying `lt_map_fn` externally
        (or via `_to_full_grid` on this class) for indexing into events.
    """
    base_ICAR_class = initialize_ICAR(dimension, length_scales=False)
    from ..models.car import lower_triangular_map

    lt_map_fn = lower_triangular_map(bins_first)
    tri_size = bins_first * (bins_first + 1) // 2

    class ICAR_normalized_LT(base_ICAR_class):
        """
        Lower-triangular variant of the normalized ICAR distribution.

        The sampled vector lives on the unique LT entries; `lt_map_fn` is
        applied internally to evaluate the quadratic form on the full
        mirrored grid.
        """

        arg_constraints = {
            "single_dimension_adj_matrices": constraints.independent(
                constraints.dependent(is_discrete=False, event_dim=2), 1
            ),
            "dx": constraints.positive,
        }
        reparametrized_params = ["single_dimension_adj_matrices", "dx"]
        pytree_aux_fields = ("is_sparse", "single_dimension_adj_matrices", "dx")

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
                for adj in single_dimension_adj_matrices:
                    if adj.ndim != 2:
                        raise ValueError(
                            "Currently, we only support 2-dimensional adj_matrix."
                        )
                    if not (
                        isinstance(adj, np.ndarray) or _is_sparse(adj)
                    ):
                        raise ValueError(
                            "adj_matrix must be a numpy array or a scipy sparse matrix."
                        )
                    self.single_dimension_adj_matrices.append(_to_sparse(adj))
            else:
                for adj in single_dimension_adj_matrices:
                    assert not _is_sparse(adj), (
                        "single_dimension_adj_matrix is sparse so please specify is_sparse=True."
                    )
                    (adj,) = promote_shapes(
                        adj, shape=batch_shape + adj.shape[-2:]
                    )
                    self.single_dimension_adj_matrices.append(adj)

            full_shape = tuple(
                [jnp.shape(mat)[-1] for mat in self.single_dimension_adj_matrices]
            )
            assert (
                full_shape[0] == bins_first and full_shape[1] == bins_first
            ), (
                f"first two adjacency matrices must have size bins_first={bins_first}; "
                f"got {full_shape[:2]}"
            )
            extra_shape = tuple(full_shape[2:])
            event_shape = (tri_size,) + extra_shape

            Distribution.__init__(
                self,
                batch_shape=batch_shape,
                event_shape=event_shape,
                validate_args=validate_args,
            )

        @property
        def support(self) -> constraints.Constraint:
            return _LogSimplexNDim(
                logsumexp=-jnp.log(self.dx), event_shape=self.event_shape
            )

        def sample(self, key, sample_shape=()):
            raise NotImplementedError

        def _to_full_grid(self, phi_LT):
            """Apply lt_map to convert LT vector (+ extras) to full mirrored grid."""
            return lt_map_fn(phi_LT)

        @validate_sample
        def log_prob(self, phi_LT):
            phi_full = self._to_full_grid(phi_LT)
            # Effective DOF: tri_size LT entries minus 1 LSE constraint, times any
            # trailing dims that are NOT in the LT pair.
            n_dof = int(np.prod(self.event_shape)) - 1
            lp, _ = _lt_normalized_log_prob_and_log_quad(
                phi_full, self.single_dimension_adj_matrices, dimension, n_dof
            )
            return lp

        def log_prob_and_quad(self, phi_LT):
            """Compute log-prior factor and quadratic form together."""
            phi_full = self._to_full_grid(phi_LT)
            n_dof = int(np.prod(self.event_shape)) - 1
            return _lt_normalized_log_prob_and_log_quad(
                phi_full, self.single_dimension_adj_matrices, dimension, n_dof
            )

        def quad_form(self, phi_LT):
            """
            Compute only the quadratic form phi^T Q phi (no eigenvalues, no logdet).

            The quadratic form is evaluated on the full mirrored grid obtained by
            applying lt_map to the LT vector. Used to recover sigma samples via
            the conditional Gamma distribution.
            """
            phi_full = self._to_full_grid(phi_LT)
            prec_mat = []
            for ii, adj in enumerate(self.single_dimension_adj_matrices):
                if self.is_sparse:
                    D = np.asarray(adj.sum(axis=-1)).squeeze(axis=-1)
                    scaled_single_prec = np.diag(D) - adj.toarray()
                else:
                    D = adj.sum(axis=-1)
                    scaled_single_prec = jnp.diag(D) - adj
                prec_mat.append(jnp.asarray(scaled_single_prec))

            logquad = 0.0
            for ii in range(dimension):
                z = jnp.moveaxis(
                    jnp.tensordot(
                        prec_mat[ii],
                        jnp.moveaxis(phi_full, ii, 0),
                        axes=(0, 0),
                    ),
                    0,
                    ii,
                )
                step = jnp.tensordot(z, phi_full, axes=dimension)
                logquad += step

            return logquad

        @staticmethod
        def infer_shapes(single_dimension_adj_matrices):
            full = tuple(
                [jnp.shape(mat)[-1] for mat in single_dimension_adj_matrices]
            )
            event_shape = (tri_size,) + tuple(full[2:])
            batch_shape = lax.broadcast_shapes(
                *[jnp.shape(mat)[:-2] for mat in single_dimension_adj_matrices]
            )
            return batch_shape, event_shape

    return ICAR_normalized_LT