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
from ..models.car import ICAR_length_scales, add_outer, mult_outer

from numpyro.distributions.transforms import (
    ComposeTransform,
    StickBreakingTransform, 
    ExpTransform, 
    AffineTransform
)


class sigma_marginalized_ICAR(ICAR_length_scales):

    def __init__(
        self, 
        single_dimension_adj_matrices,
        *args,
        is_sparse=False,
        validate_args=None,
    ):
        base_lsigma = 0.
        super(sigma_marginalized_ICAR, self).__init__(
            base_lsigma,
            single_dimension_adj_matrices,
            *args,
            is_sparse=is_sparse,
            validate_args=validate_args,
        )
    
    @validate_sample
    def log_prob(self, phi):
        return self.log_prob_and_quad(phi)[0]
    
    def log_prob_and_quad(self, phi):
        # same as log_prob, but also returns logquad for producing sigma (coupling) samples from conditional Gamma distribution

        lams = []
        prec_mat = []
        n = 1
        for ii, single_dimension_adj_matrix in enumerate(self.single_dimension_adj_matrices):
            if self.is_sparse:
                D = np.asarray(single_dimension_adj_matrix.sum(axis=-1)).ravel()
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
                # print(jnp.diag(D).shape, single_dimension_adj_matrix.shape)
                lam = jnp.linalg.eigvalsh(scaled_single_prec)
                lam = lam.at[0].set(0.) # set to zero, otherwise float precision can allow this to be negative and cause problems
            prec_mat.append(jnp.asarray(scaled_single_prec))
            lams.append(lam)

        if len(lams) > 1:
            ar = reduce(add_outer, lams)
        else:
            ar = lams[0]
        if isinstance(ar, np.ndarray):
            ar[(0,)*self.dimension] = 1.
        else:
            ar = ar.at[(0,)*self.dimension].set(1.)
        logdet = jnp.sum(jnp.log(ar))
        
        logquad = 0.
        for ii in range(self.dimension):
            z = jnp.moveaxis(
                jnp.tensordot(prec_mat[ii], jnp.moveaxis(phi,ii,0), axes=(0,0)), # tensordot requires a concrete axis ... can I get this in a jitted fn?
                0,ii)
            
            step = jnp.tensordot(z, phi, axes=self.dimension)
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


class StudentICAR(Distribution):
    
    arg_constraints = {
        "nu": constraints.real, 
        "log_sigmas": constraints.real_vector, 
        "single_dimension_adj_matrices": constraints.independent(constraints.dependent(is_discrete=False, event_dim=2), 1),
        "dof_correction": constraints.positive, 
    }
    reparametrized_params = [
        "nu",
        "log_sigmas",
        "single_dimension_adj_matrices",
        "dof_correction",
    ]
    pytree_aux_fields = ("is_sparse", "single_dimension_adj_matrices")

    def __init__(
        self,
        nu,
        log_sigmas,
        single_dimension_adj_matrices,
        *,
        dof_correction=1,
        is_sparse=False,
        validate_args=None,
    ):
        self.nu = nu
        self.dof_correction = dof_correction
        self.dimension = len(single_dimension_adj_matrices)
        
        if jnp.ndim(log_sigmas) == 1:
            self.log_sigmas = log_sigmas
        elif jnp.ndim(log_sigmas) == 0:
            self.log_sigmas = lax.broadcast(log_sigmas, (self.dimension,))
        assert self.log_sigmas.shape[0] == self.dimension
        
        self.is_sparse = is_sparse
        batch_shape = ()
        
        self.single_dimension_adj_matrices = []
        self.edges = [] 
        
        if self.is_sparse:
            for single_dimension_adj_matrix in single_dimension_adj_matrices:
                if single_dimension_adj_matrix.ndim != 2:
                    raise ValueError("Currently, we only support 2-dimensional adj_matrix.")
                if not (isinstance(single_dimension_adj_matrix, np.ndarray) or _is_sparse(single_dimension_adj_matrix)):
                    raise ValueError("adj_matrix needs to be a numpy array or a scipy sparse matrix.")
                
                sparse_mat = _to_sparse(single_dimension_adj_matrix)
                self.single_dimension_adj_matrices.append(sparse_mat)
                u, v = np.where(np.triu(sparse_mat.toarray()) > 0)
                self.edges.append((jnp.array(u), jnp.array(v)))
        else:
            for single_dimension_adj_matrix in single_dimension_adj_matrices:
                assert not _is_sparse(single_dimension_adj_matrix), "single_dimension_adj_matrix is sparse; specify `is_sparse=True`."
                
                (single_dimension_adj_matrix,) = promote_shapes(
                    single_dimension_adj_matrix, shape=batch_shape + single_dimension_adj_matrix.shape[-2:]
                )
                self.single_dimension_adj_matrices.append(single_dimension_adj_matrix)
                dense_mat = np.asarray(single_dimension_adj_matrix)
                u, v = np.where(np.triu(dense_mat) > 0)
                self.edges.append((jnp.array(u), jnp.array(v)))

        event_shape = tuple([jnp.shape(mat)[-1] for mat in self.single_dimension_adj_matrices])

        super(StudentICAR, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=False,
        )

    def sample(self, key, sample_shape=()):
        raise NotImplementedError 

    @property
    def support(self):
        return constraints.independent(constraints.real, self.dimension)

    @validate_sample
    def log_prob(self, phi):
        precs = jnp.exp(-2.0 * self.log_sigmas)
        
        # 1. Precompute expensive terms and constants outside the loop
        half_nu = 0.5 * self.nu
        half_nu_plus_half = half_nu + 0.5
        
        log_gamma_term = (
            gammaln(half_nu_plus_half) 
            - gammaln(half_nu) 
            - 0.5 * jnp.log(self.nu * jnp.pi)
        )
        
        log_prob_total = 0.0
        total_edges_evaluated = 0
        v_nodes = float(np.prod(self.event_shape))
        
        # 2. Use a static tuple for event axes
        event_axes = tuple(range(-self.dimension, 0))

        for ii, (u, v) in enumerate(self.edges):
            axis = -self.dimension + ii
            
            # 3. Gather adjacent slices
            phi_u = jnp.take(phi, u, axis=axis)
            phi_v = jnp.take(phi, v, axis=axis)
            
            # 4. Use jnp.square for slightly faster XLA compilation
            sq_diffs = jnp.square(phi_u - phi_v)
            
            tau = precs[ii]
            log_term = jnp.log1p(tau * sq_diffs / self.nu)
            quad_term = jnp.sum(log_term, axis=event_axes)
            
            # 5. Compute static edge counts using Python integers
            n_edges = 1
            for ax_idx in range(-self.dimension, 0):
                if ax_idx == axis:
                    n_edges *= int(u.shape[0])
                else:
                    n_edges *= int(self.event_shape[ax_idx])
            
            total_edges_evaluated += n_edges
            
            norm_const = n_edges * (log_gamma_term + 0.5 * jnp.log(tau))
            log_prob_total += norm_const - half_nu_plus_half * quad_term
            
        dof_correction = (v_nodes * self.dof_correction - 1.0) / float(total_edges_evaluated) if total_edges_evaluated > 0 else 1.0

        return log_prob_total * dof_correction
    
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
            
        d.edges = []
        for mat in d.single_dimension_adj_matrices:
            dense_adj = mat.toarray() if d.is_sparse else np.asarray(mat)
            u, v = np.where(np.triu(dense_adj) > 0)
            d.edges.append((jnp.array(u), jnp.array(v)))
            
        return d

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


class grid_marginalized_ICAR_length_scales(ICAR_length_scales):
    """
    ICAR distribution with per-dimension length scales numerically
    marginalized over an ND grid.

    For a D-dimensional ICAR model with independent length scales
    ``(sigma_1, ..., sigma_D)``, this distribution computes:

    .. math::
        p(\\phi) = \\int p(\\phi \\mid \\sigma) \\, p(\\sigma) \\, d\\sigma

    by evaluating ``p(phi | sigma) * p(sigma)`` on a Cartesian grid of
    ``log(sigma)`` values and integrating via logsumexp (trapezoidal rule
    in log-space).

    Parameters
    ----------
    single_dimension_adj_matrices : list of ndarray or sparse matrices
        List of adjacency matrices, one per spatial dimension.
    lnsigma_ranges : list of (float, float)
        Per-dimension ``(min, max)`` bounds for the ``log(sigma)`` grid.
        If a single tuple is given, it is broadcast to all dimensions.
    grid_points : int or list of int
        Number of grid points per dimension. If a single int, broadcast
        to all dimensions. Default 50.
    log_prior_fn : callable or None
        Function mapping a ``log(sigma)`` array of shape ``(D,)`` to a
        scalar log-prior value. If ``None`` (default), uses a flat prior
        in ``log(sigma)`` (Jeffreys prior on ``sigma``).
    is_sparse : bool, optional
        Whether the adjacency matrices are sparse (default False).

    Notes
    -----
    Memory scales as ``prod(grid_points) * prod(bins_per_dim)``. For large
    problems, reduce ``grid_points`` accordingly.
    """

    def __init__(
        self,
        single_dimension_adj_matrices,
        lnsigma_ranges,
        grid_points=50,
        log_prior_fn=None,
        *,
        is_sparse=False,
        validate_args=None,
    ):
        # Use a dummy log_sigma=0 to initialize the parent class
        # (eigenvalues, precision matrices, shapes, etc.)
        base_lsigma = 0.
        super().__init__(
            base_lsigma,
            single_dimension_adj_matrices,
            is_sparse=is_sparse,
            validate_args=validate_args,
        )

        dimension = self.dimension

        # Parse grid specification
        if isinstance(lnsigma_ranges, tuple) and len(lnsigma_ranges) == 2 and not isinstance(lnsigma_ranges[0], tuple):
            lnsigma_ranges = [lnsigma_ranges] * dimension
        assert len(lnsigma_ranges) == dimension

        if isinstance(grid_points, int):
            grid_points = [grid_points] * dimension
        assert len(grid_points) == dimension

        # Build 1D grids and volume element
        self._grids_1d = []
        log_dvol = 0.
        for i in range(dimension):
            lo, hi = lnsigma_ranges[i]
            g = jnp.linspace(lo, hi, grid_points[i])
            self._grids_1d.append(g)
            if grid_points[i] > 1:
                log_dvol += jnp.log(g[1] - g[0])

        self._log_dvol = log_dvol

        # Precompute 1D eigenvalues (unscaled by precision)
        self._eigenvalues_1d = []
        self._prec_mats = []
        self._n = 1
        for ii, single_dimension_adj_matrix in enumerate(single_dimension_adj_matrices):
            if is_sparse:
                D = np.asarray(single_dimension_adj_matrix.sum(axis=-1)).ravel()
                scaled_single_prec = np.diag(D) - single_dimension_adj_matrix.toarray()
            else:
                D = single_dimension_adj_matrix.sum(axis=-1)
                scaled_single_prec = jnp.diag(D) - single_dimension_adj_matrix

            self._n *= D.shape[-1]

            if isinstance(scaled_single_prec, np.ndarray):
                lam = np.linalg.eigvalsh(scaled_single_prec)
                lam[0] = 0.
            else:
                lam = jnp.linalg.eigvalsh(scaled_single_prec)
                lam = lam.at[0].set(0.)

            self._eigenvalues_1d.append(jnp.asarray(lam))
            self._prec_mats.append(jnp.asarray(scaled_single_prec))

        # Precompute ND grid of precisions: shape (G1, ..., GD, D)
        # and log-prior values: shape (G1, ..., GD)
        meshes = jnp.meshgrid(*self._grids_1d, indexing='ij')
        # lnsigma_grid: shape (G1, ..., GD, D)
        lnsigma_grid = jnp.stack(meshes, axis=-1)
        # prec_grid: shape (G1, ..., GD, D)
        self._prec_grid = jnp.exp(-2. * lnsigma_grid)

        if log_prior_fn is not None:
            # Evaluate log-prior at every grid point
            flat_lnsigma = lnsigma_grid.reshape(-1, dimension)
            flat_lp = jnp.array([log_prior_fn(flat_lnsigma[j]) for j in range(flat_lnsigma.shape[0])])
            self._log_prior_grid = flat_lp.reshape(lnsigma_grid.shape[:-1])
        else:
            # Flat prior in log-sigma (Jeffreys on sigma)
            self._log_prior_grid = jnp.zeros(lnsigma_grid.shape[:-1])

        # Precompute the logdet contribution for every grid point.
        # eigenvalue array at grid point g: Lambda[k1,...,kD] = sum_i prec_g[i] * lam_i[ki]
        # We build this via broadcasting.
        # For each dimension i, create an array of shape
        #   (1,...,Gi,...,1, 1,...,Ni,...,1)
        #          ^ grid      ^ eigenvalue
        # with Gi at position i (grid dims) and Ni at position dimension+i (eigenvalue dims).
        total_ndim = 2 * dimension  # D grid dims + D eigenvalue dims
        terms = []
        for i in range(dimension):
            prec_shape = [1] * total_ndim
            prec_shape[i] = grid_points[i]
            # Extract the 1D precision values for dimension i: shape (Gi,)
            # Extract 1D precision along grid axis i: _prec_grid has shape (G1,...,GD, D)
            # We want _prec_grid[:,...,0,...,0, i] with slice(None) at position i
            idx = [0] * (dimension + 1)
            idx[i] = slice(None)
            idx[-1] = i
            prec_1d = self._prec_grid[tuple(idx)]

            lam_shape = [1] * total_ndim
            lam_shape[dimension + i] = self._eigenvalues_1d[i].shape[0]

            term = prec_1d.reshape(prec_shape) * self._eigenvalues_1d[i].reshape(lam_shape)
            terms.append(term)

        full_lam = sum(terms)  # shape (G1,...,GD, N1,...,ND)
        # Fix the zero eigenvalue: at eigenvalue index (0,...,0), set to sum of precs
        zero_eig_idx = (slice(None),) * dimension + (0,) * dimension
        sum_precs = self._prec_grid.sum(axis=-1)  # shape (G1,...,GD)
        full_lam = full_lam.at[zero_eig_idx].set(sum_precs)

        # logdet: sum over eigenvalue dimensions
        eig_axes = tuple(range(dimension, total_ndim))
        self._logdet_grid = jnp.sum(jnp.log(full_lam), axis=eig_axes)
        # shape (G1,...,GD)

    @validate_sample
    def log_prob(self, phi):
        dimension = self.dimension
        n = self._n

        # Compute per-dimension quadratic forms: q_i = phi^T Q_i phi
        quads = []
        for ii in range(dimension):
            z = jnp.moveaxis(
                jnp.tensordot(self._prec_mats[ii], jnp.moveaxis(phi, ii, 0), axes=(0, 0)),
                0, ii,
            )
            quads.append(jnp.tensordot(z, phi, axes=dimension))

        # quadform at each grid point: sum_i prec_grid[..., i] * q_i
        # shape (G1, ..., GD)
        quads_arr = jnp.stack(quads)  # (D,)
        logquad_grid = jnp.sum(self._prec_grid * quads_arr, axis=-1)

        # log p(phi | sigma) at each grid point
        log_p_grid = 0.5 * (-n * jnp.log(2 * jnp.pi) + self._logdet_grid - logquad_grid)

        # Marginalize: logsumexp over grid + log(volume element) + log-prior
        return LSE(log_p_grid + self._log_prior_grid) + self._log_dvol

    def log_prob_and_conditional_lnsigma(self, phi):
        """
        Compute the marginal log-prob and the conditional posterior over
        log(sigma) given phi.

        Returns
        -------
        log_prob : scalar
            The grid-marginalized log-probability.
        conditional_log_weights : jnp.ndarray, shape (G1, ..., GD)
            Log-posterior weights for each grid point (unnormalized).
            Useful for drawing conditional sigma samples or diagnostics.
        """
        dimension = self.dimension
        n = self._n

        quads = []
        for ii in range(dimension):
            z = jnp.moveaxis(
                jnp.tensordot(self._prec_mats[ii], jnp.moveaxis(phi, ii, 0), axes=(0, 0)),
                0, ii,
            )
            quads.append(jnp.tensordot(z, phi, axes=dimension))

        quads_arr = jnp.stack(quads)
        logquad_grid = jnp.sum(self._prec_grid * quads_arr, axis=-1)

        log_p_grid = 0.5 * (-n * jnp.log(2 * jnp.pi) + self._logdet_grid - logquad_grid)
        log_joint = log_p_grid + self._log_prior_grid

        log_marg = LSE(log_joint) + self._log_dvol
        return log_marg, log_joint

    @staticmethod
    def infer_shapes(single_dimension_adj_matrices):
        event_shape = tuple([jnp.shape(mat)[-1] for mat in single_dimension_adj_matrices])
        batch_shape = lax.broadcast_shapes(
            *[jnp.shape(mat)[:-2] for mat in single_dimension_adj_matrices]
        )
        return batch_shape, event_shape


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
