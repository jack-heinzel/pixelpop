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

from functools import reduce

def add_outer(a, b):
    a_exp = jnp.expand_dims(a, axis=tuple(range(a.ndim,b.ndim+a.ndim)))
    b_exp = jnp.expand_dims(b, axis=tuple(range(0,a.ndim)))
    return a_exp + b_exp

def mult_outer(a, b):
    a_exp = jnp.expand_dims(a, axis=tuple(range(a.ndim,b.ndim+a.ndim)))
    b_exp = jnp.expand_dims(b, axis=tuple(range(0,a.ndim)))
    return a_exp * b_exp

def initialize_ICAR(dimension, length_scales=False):
    """
    Construct an Intrinsic Conditional Autoregressive (ICAR) distribution class.

    The returned class defines a NumPyro-compatible ICAR prior with optional
    length-scale parameters. The ICAR is a special case of a Gaussian Markov
    random field where the precision matrix is determined by adjacency
    matrices of spatial sites.

    Parameters
    ----------
    dimension : int
        Number of spatial dimensions.
    length_scales : bool, optional (default=False)
        If True, allows dimension-specific log-scale parameters. If False,
        a single log-scale parameter is shared.

    Returns
    -------
    ICAR_length_scales : class
        A NumPyro `Distribution` subclass implementing the ICAR prior, with
        methods for `log_prob`, shape inference, and JAX pytree compatibility.

    Notes
    -----
    - The distribution is improper (cannot sample directly).
    - Adjacency matrices must be symmetric with all sites having neighbors.
    - Sparse and dense adjacency matrix representations are supported.
    """

    class _RealTensor(constraints._IndependentConstraint, constraints._SingletonConstraint):
        def __init__(self):
            super().__init__(constraints._Real(), dimension)

    realtensor = _RealTensor()
    biject_to = numpyro.distributions.transforms.biject_to

    @biject_to.register(type(realtensor))
    @biject_to.register(constraints.independent)
    def _biject_to_independent(constraint):
        return numpyro.distributions.transforms.IndependentTransform(
            biject_to(constraint.base_constraint), constraint.reinterpreted_batch_ndims
        )

    class ICAR_length_scales(Distribution):
        """
        Intrinsic Conditional Autoregressive (ICAR) distribution with optional
        length-scale parameters for each spatial dimension.

        The ICAR distribution is a Gaussian Markov random field with a precision
        matrix determined by adjacency matrices of sites.

        The log_prob function is a minor modification of Eq. 18 in https://arxiv.org/abs/2406.16813
        The model as written here is described here:
        https://git.ligo.org/jack.heinzel/pixelpop/-/wikis/pixelpop-generic-review#is-the-intrinsic-conditional-autoregressive-icar-model-correctly-implemented

        Parameters
        ----------
        log_sigmas : array-like, shape (dimension,) or scalar
            Logarithm of the standard deviation(s) of the ICAR prior. If `length_scales`
            is True, can provide a separate log_sigma for each dimension; otherwise,
            a single log_sigma is broadcast to all dimensions.
        single_dimension_adj_matrices : list of ndarray or sparse matrices
            List of adjacency matrices, one per spatial dimension. Each matrix must
            be symmetric, with all sites having at least one neighbor.
        is_sparse : bool, optional (default=False)
            Whether to treat the adjacency matrices as sparse. Must be True if the
            adjacency matrices are `scipy.sparse` objects.
        validate_args : bool, optional
            Whether to validate input arguments.

        Attributes
        ----------
        log_sigmas : jnp.ndarray
            Broadcasted log-scale parameters for each dimension.
        single_dimension_adj_matrices : list
            List of adjacency matrices (sparse or dense) for each dimension.
        is_sparse : bool
            Indicates whether sparse operations are used.

        Methods
        -------
        log_prob(phi)
            Compute the log-probability of the field `phi` under the ICAR prior.
        sample(key, sample_shape=())
            Not implemented; ICAR is improper and cannot be sampled directly.
        infer_shapes(log_sigmas, single_dimension_adj_matrices)
            Utility method to infer batch and event shapes.
        tree_flatten()
            Support for JAX pytree flattening.
        tree_unflatten(aux_data, params)
            Support for JAX pytree unflattening.

        Notes
        -----
        - The ICAR prior is improper; its log-probability is defined up to a constant.
        - Sparse adjacency matrices can be used for efficiency but must be symmetric.
        - The distribution is suitable as a prior in hierarchical models.
        """

        arg_constraints = {
            "log_sigmas": constraints.real_vector, # vector of length dimension
            "single_dimension_adj_matrices": constraints.independent(constraints.dependent(is_discrete=False, event_dim=2), 1),
        }
        support = realtensor
        reparametrized_params = [
            "log_sigmas",
            "single_dimension_adj_matrices",
        ]
        pytree_aux_fields = ("is_sparse", "adj_matrix")

        def __init__(
            self,
            log_sigmas,
            single_dimension_adj_matrices,
            *,
            is_sparse=False,
            validate_args=None,
        ):
            if length_scales:
                if jnp.ndim(log_sigmas) == 0:
                    assert dimension == 1
                    (log_sigmas,) = promote_shapes(log_sigmas, shape=(1,))
            else:
                assert jnp.ndim(log_sigmas) == 0
                (log_sigmas,) = promote_shapes(log_sigmas, shape=(dimension,))
            self.is_sparse = is_sparse

            batch_shape = ()
            # print('batch shape is ', batch_shape)
            self.single_dimension_adj_matrices = []
            if self.is_sparse:
                for single_dimension_adj_matrix in single_dimension_adj_matrices:
                    if single_dimension_adj_matrix.ndim != 2:
                        raise ValueError(
                            "Currently, we only support 2-dimensional adj_matrix. Please make a feature request",
                            " if you need higher dimensional adj_matrix.",
                        )
                    if not (isinstance(single_dimension_adj_matrix, np.ndarray) or _is_sparse(single_dimension_adj_matrix)):
                        raise ValueError(
                            "adj_matrix needs to be a numpy array or a scipy sparse matrix. Please make a feature",
                            " request if you need to support jax ndarrays.",
                        )
                    # TODO: look into future jax sparse csr functionality and other developments
                    self.single_dimension_adj_matrices.append(_to_sparse(single_dimension_adj_matrix))
            else:
                for single_dimension_adj_matrix in single_dimension_adj_matrices:
                    assert not _is_sparse(single_dimension_adj_matrix), (
                        "single_dimension_adj_matrix is a sparse matrix so please specify `is_sparse=True`."
                    )
                    # TODO: look into static jax ndarray representation
                    (single_dimension_adj_matrix,) = promote_shapes(
                        single_dimension_adj_matrix, shape=batch_shape + single_dimension_adj_matrix.shape[-2:]
                    )
                    self.single_dimension_adj_matrices.append(single_dimension_adj_matrix)

            event_shape = tuple([jnp.shape(mat)[-1] for mat in self.single_dimension_adj_matrices])

            (self.log_sigmas,) = promote_shapes(log_sigmas, shape=batch_shape + log_sigmas.shape[-1:])

            super(ICAR_length_scales, self).__init__(
                batch_shape=batch_shape,
                event_shape=event_shape,
                validate_args=validate_args,
            )

            for single_dimension_adj_matrix in self.single_dimension_adj_matrices:
                if self._validate_args and (isinstance(single_dimension_adj_matrix, np.ndarray) or is_sparse):
                    assert (single_dimension_adj_matrix.sum(axis=-1) > 0).all() > 0, (
                        "all sites in adjacency matrix must have neighbours"
                    )

                    if self.is_sparse:
                        assert (single_dimension_adj_matrix != single_dimension_adj_matrix.T).nnz == 0, (
                            "adjacency matrix must be symmetric"
                        )
                    else:
                        assert np.array_equal(
                            single_dimension_adj_matrix, np.swapaxes(single_dimension_adj_matrix, -2, -1)
                        ), "adjacency matrix must be symmetric"

        def sample(self, key, sample_shape=()):
            # cannot sample from an improper distribution
            raise NotImplementedError 

        @validate_sample
        def log_prob(self, phi):

            precs = jnp.exp(-2*self.log_sigmas)
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
                lams.append(precs[ii]*lam)

            ar = reduce(add_outer, lams)
            # print(ar.shape)
            logdet = jnp.sum(jnp.log(ar.at[(0,)*dimension].set(jnp.sum(precs))))
            logquad = 0.
            for ii in range(dimension):
            #def step_fn(ii):
                z = jnp.moveaxis(
                    jnp.tensordot(prec_mat[ii], jnp.moveaxis(phi,ii,0), axes=(0,0)), # tensordot requires a concrete axis ... can I get this in a jitted fn?
                    0,ii)
                # print(z.shape,y.shape,x.shape)
                step = jnp.tensordot(z, phi, axes=dimension)
                logquad += step * precs[ii]
                #return step

            # kronecker_terms = jax.vmap(step_fn)(jnp.arange(dimension, dtype=int))
            # logquad = jnp.dot(kronecker_terms, precs)
            
            return 0.5 * (-n * jnp.log(2*jnp.pi) + logdet - logquad)

        @staticmethod
        def infer_shapes(log_sigmas, single_dimension_adj_matrices):
            event_shape = tuple([jnp.shape(mat)[-1] for mat in single_dimension_adj_matrices])
            batch_shape = lax.broadcast_shapes(
                jnp.shape(log_sigmas)[:-1], *[jnp.shape(mat)[:-2] for mat in single_dimension_adj_matrices]
            )
            return batch_shape, event_shape

        def tree_flatten(self):
            data, aux = super().tree_flatten()
            single_dimension_adj_matrix_data_idx = type(self).gather_pytree_data_fields().index("single_dimension_adj_matrices")
            single_dimension_adj_matrix_aux_idx = type(self).gather_pytree_aux_fields().index("single_dimension_adj_matrices")

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
                adj_matrix_data_idx = cls.gather_pytree_data_fields().index("single_dimension_adj_matrices")
                setattr(d, "single_dimension_adj_matrices", params[adj_matrix_data_idx])
            else:
                adj_matrix_aux_idx = cls.gather_pytree_aux_fields().index("single_dimension_adj_matrices")
                setattr(d, "single_dimension_adj_matrices", aux_data[adj_matrix_aux_idx])
            return d

    return ICAR_length_scales

def lower_triangular_map(bins):
    """
    Build a mapping from vectorized lower-triangular indices to a full symmetric matrix.

    Given the number of bins (matrix size), this returns a function that reconstructs
    a symmetric matrix from an array storing only the unique lower-triangular entries.

    Parameters
    ----------
    bins : int
        Size of the symmetric matrix (number of rows/columns).

    Returns
    -------
    symmetric_from_tri : function
        A function that takes an array of lower-triangular elements and
        returns a symmetric `(bins, bins, ...)` array.

    Examples
    --------
    >>> sym_from_tri = lower_triangular_map(3)
    >>> arr = jnp.arange(6)  # lower-triangular entries
    >>> sym_from_tri(arr).shape
    (3, 3)
    """
    sym_shape = jnp.array([bins,bins])
    a, b = jnp.unravel_index(jnp.arange(bins**2), sym_shape)
    a, b = jnp.minimum(a, b), jnp.maximum(a, b)
    map_arr = jnp.array((bins - (a+1)/2)*a + b, dtype=int)

    def symmetric_from_tri(arr):
        s = arr.shape
        return arr[map_arr,...].reshape((bins, bins)+s[1:])
    
    return symmetric_from_tri

def lower_triangular_log_prob(phi, n, log_sigma, single_dimension_adj_matrices):
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

    prec = jnp.exp(-2*log_sigma) 
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
        logquad += step * prec
    
    return 0.5 * (-n * jnp.log(2*jnp.pi) - 2 * n * log_sigma - logquad)

def axes_tril(array, axes=(-2,-1)):
    '''
    wrapper for np.tril, and allows for custom axes to take the 
    lower triangular components of
    '''
    array = np.moveaxis(array, axes, (-2,-1))
    array = np.tril(array)
    array = np.moveaxis(array, (-2,-1), axes)
    return array