import numpy as np
from .gwpop_models import * 
from .car import ICAR_length_scales, lower_triangular_log_prob, lower_triangular_map
from ..experimental.car import (
    DiagonalizedICARTransform,
    MaternSPDETransform,
    WKBNonStationaryMaternSPDETransform,
    StudentICAR,
    sigma_marginalized_ICAR,
    grid_marginalized_ICAR_length_scales,
    lower_triangular_sigma_marg_log_prob,
    lower_triangular_sigma_marg_log_prob_and_log_quad
)
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.debug import print as jaxprint
from ..utils.data import place_in_bins
from jax.scipy.special import logsumexp as LSE
import numpyro
from numpyro.infer import MCMC, NUTS
from tqdm import tqdm
import sys
from numpyro.diagnostics import (effective_sample_size, print_summary,
                                 split_gelman_rubin)
from jax import random
import os
from contextlib import redirect_stdout
import h5ify
from numpyro import handlers
from functools import partial
from .reparameterization import (
    ordered_pair_bounds,
    ordered_pair_initial_value,
    reparameterized_sites,
    sample_ordered_pair,
)

def setup_probabilistic_model(pixelpop_data, log='default'):
    """
    Construct a hierarchical probabilistic model for GW population inference.

    This function sets up both parametric and nonparametric (CAR/ICAR) components
    of a gravitational-wave population model, returning a NumPyro-compatible model
    along with suitable initial values for MCMC warmup.

    
    Returns
    -------
    probabilistic_model : callable
        NumPyro-compatible probabilistic model.
    initial_value : dict
        Suggested initial values for MCMC warmup.
    """
    
    if pixelpop_data.lower_triangular:
        lt_map = lower_triangular_map(pixelpop_data.bins[0])
        tri_size = int(pixelpop_data.bins[0]*(pixelpop_data.bins[0]+1)/2) 
        unique_sample_shape = (tri_size,) + tuple(pixelpop_data.bins[2:])
        normalization_dof = tri_size * int(np.prod(pixelpop_data.bins[2:])) # lower triangular in first two dimensions
    else:
        normalization_dof = int(np.prod(pixelpop_data.bins))
    def get_initial_value(plausible_hyperparameters, parameters, Nobs, inj_weights, random_initialization):
        """
        Construct initial values for the pixelized (nonparametric) merger rate density.

        Parameters
        ----------
        plausible_hyperparameters : dict
            Plausible hyperparameter values used for initialization if not random.
        parameters : list of str
            Parameters included in the nonparametric model.
        Nobs : int
            Number of observed events.
        inj_weights : ndarray
            Logarithmic weights from injections, adjusted for prior volume.
        random_initialization : bool
            If True, initialize randomly; otherwise use plausible hyperparameters.

        Returns
        -------
        initial_value : dict
            Dictionary containing initial 'merger_rate_density' or 'base_interpolation'.
        """
        bin_med = [
            (pixelpop_data.bin_axes[ii][:-1] + pixelpop_data.bin_axes[ii][1:])/2 
            for ii in range(pixelpop_data.dimension)
            ]
        interpolation_grid = np.meshgrid(*bin_med, indexing='ij')

        # When sampling in the Gaussian IID eigenbasis, the free sites are the
        # standard-normal '_eigenbasis_sites' (shape = bins) plus a scalar
        # 'log_rate_offset' carrying the overall rate; merger_rate_density is a
        # deterministic of these and cannot be initialized directly.
        if pixelpop_data.diagonalize_icar:
            eigenbasis_init = jnp.array(
                np.random.normal(loc=0, scale=1, size=pixelpop_data.bins)
                )
            if random_initialization:
                log_rate_offset_init = 0.0
            else:
                # Seed the offset near the expected log rate, mirroring the Rexp
                # computation in the standard (non-eigenbasis) initialization.
                data_grid = {p.replace('_psi',''): interpolation_grid[ii] for ii, p in enumerate(parameters)}
                initial_interpolation = np.sum([
                    pixelpop_data.parametric_models[p](data_grid, *[
                        plausible_hyperparameters[h]
                        for h in pixelpop_data.parameter_to_hyperparameters[p]
                        ])
                    for ii, p in enumerate(parameters)
                ], axis=0)
                pdet = LSE(initial_interpolation[pixelpop_data.inj_bins] + inj_weights) - jnp.log(pixelpop_data.injections['total_generated'])
                log_rate_offset_init = jnp.log(Nobs) - pdet - jnp.log(pixelpop_data.injections['analysis_time'])
            init = {
                '_eigenbasis_sites': eigenbasis_init,
                'log_rate_offset': jnp.asarray(log_rate_offset_init, dtype=float),
                }
            if pixelpop_data.spde_wkb:
                dim = pixelpop_data.dimension
                init['log_nu_spde'] = jnp.asarray(0.0)
                init['log_ranges'] = jnp.log(jnp.asarray(pixelpop_data.bins, dtype=float) / 4.)
                # Start at the stationary background: all slopes zero.
                init['range_response'] = jnp.zeros((dim, dim))
                init['nu_slope'] = jnp.zeros(dim)
                init['sigma_slope'] = jnp.zeros(dim)
            elif pixelpop_data.spde_matern:
                init['nu_spde'] = jnp.asarray(1.0)
                init['log_ranges'] = jnp.log(jnp.asarray(pixelpop_data.bins, dtype=float) / 4.)
            return init

        return_key = 'merger_rate_density'
        if random_initialization:
            if pixelpop_data.lower_triangular:
                return_dict = {'base_interpolation': jnp.array(
                    np.random.normal(loc=0, scale=1, size=unique_sample_shape)
                    )}
            else:
                return_dict = {return_key: jnp.array(
                    np.random.normal(loc=0, scale=1, size=interpolation_grid[0].shape))
                    }
                
        else:
            data_grid = {p.replace('_psi',''): interpolation_grid[ii] for ii, p in enumerate(parameters)}    
            
            initial_interpolation = np.sum([
                pixelpop_data.parametric_models[p](data_grid, *[
                    plausible_hyperparameters[h] 
                    for h in pixelpop_data.parameter_to_hyperparameters[p]
                    ]) 
                for ii, p in enumerate(parameters)
            ], axis=0)
            pdet = LSE(initial_interpolation[pixelpop_data.inj_bins] + inj_weights) - jnp.log(pixelpop_data.injections['total_generated'])
            Rexp = jnp.log(Nobs) - pdet - jnp.log(pixelpop_data.injections['analysis_time'])
            initial_interpolation = np.logaddexp(initial_interpolation, -10*np.ones_like(initial_interpolation)) # logaddexp -10 to smooth out negative divergences
            return_dict = {return_key: Rexp + initial_interpolation}
        return return_dict
            
    parameters_psi = [p.replace('redshift', 'redshift_psi') for p in pixelpop_data.pixelpop_parameters]
    if pixelpop_data.skip_nonparametric:
        initial_value = {}
    else:
        initial_value = get_initial_value(
            pixelpop_data.plausible_hyperparameters, 
            parameters_psi, 
            pixelpop_data.Nobs, 
            pixelpop_data.injections['ln_dVTc']-pixelpop_data.injections['log_prior'],
            random_initialization=pixelpop_data.random_initialization
            )

    ordered_bounds = ordered_pair_bounds(pixelpop_data.priors)
    reparameterized = {name for pair in ordered_bounds for name in pair}
    if ordered_bounds:
        initial_value.update(ordered_pair_initial_value(
            pixelpop_data.priors, pixelpop_data.plausible_hyperparameters
            ))
        if pixelpop_data.constraint_funcs:
            print(
                f"[warning] {sorted(reparameterized)} are ordered by construction; "
                "any constraint_func that also imposes that ordering now double-counts "
                f"it: {[f.__name__ for f in pixelpop_data.constraint_funcs]}"
            )

    def parametric_model(data, injections, event_weights, inj_weights):
        """
        Evaluate the parametric population model contribution.

        Draws hyperparameters from their priors and adds the corresponding
        parametric model values to the event and injection weights.

        Parameters
        ----------
        data : dict
            Event data, keyed by parameter name.
        injections : dict
            Injection data, keyed by parameter name.
        event_weights : ndarray
            Current accumulated event log-weights.
        inj_weights : ndarray
            Current accumulated injection log-weights.

        Returns
        -------
        event_weights : ndarray
            Updated event log-weights including parametric contributions.
        inj_weights : ndarray
            Updated injection log-weights including parametric contributions.
        """
        sample = {}
        for key in pixelpop_data.priors:
            if key in reparameterized:
                continue                          # drawn as an ordered pair below
            args, distribution = pixelpop_data.priors[key]
            if distribution.__name__ == 'Delta':
                sample[key] = args[0]
            else:
                sample[key] = numpyro.sample(key, distribution(*args))
        for (upper, lower), (lo, hi) in ordered_bounds.items():
            sample[upper], sample[lower] = sample_ordered_pair(upper, lower, lo, hi)

        if log == 'debug':
            for p in pixelpop_data.other_parameters:
                jaxprint('[DEBUG] =================================')
                jaxprint('[DEBUG] parametric parameters: {p}', p=p)
                jaxprint('[DEBUG] =================================')       
                for k in pixelpop_data.parameter_to_hyperparameters[p]:
                    jaxprint('[DEBUG] \t {k} sample: {s}', k=k, s=sample[k])
        for constraint_func in pixelpop_data.constraint_funcs:
            numpyro.factor(constraint_func.__name__, constraint_func(sample))
            if log == 'debug':
                jaxprint('[DEBUG] constraint functions:', constraint_func.__name__, constraint_func(sample))
        for p in pixelpop_data.other_parameters:
            event_weights += pixelpop_data.parametric_models[p](
                data, *[sample[h] for h in pixelpop_data.parameter_to_hyperparameters[p]]
                )
            inj_weights += pixelpop_data.parametric_models[p](
                injections, *[sample[h] for h in pixelpop_data.parameter_to_hyperparameters[p]]
                )
            if log == 'debug':
                jaxprint('[DEBUG] parametric {p} LSE(event_weights)={ew}, LSE(injection_weights)={iw}', p=p, ew=LSE(event_weights), iw=LSE(inj_weights))
                if not jnp.isfinite(LSE(event_weights)):
                    for parameter in pixelpop_data.parameter_to_hyperparameters[p]:
                        jaxprint('[DEBUG] inf event weights at {pp}={d}', pp=parameter, d=data[parameter][jnp.where(event_weights == jnp.inf)])
                if not jnp.isfinite(LSE(inj_weights)):
                    for parameter in pixelpop_data.parameter_to_hyperparameters[p]:
                        jaxprint('[DEBUG] inf injection weights at {pp}={d}', pp=parameter, d=injections[parameter][jnp.where(inj_weights == jnp.inf)])
        return event_weights, inj_weights

    if pixelpop_data.cauchy_icar:
        ICAR_model = StudentICAR
    elif pixelpop_data.marginalize_sigma and pixelpop_data.length_scales:
        print("[experimental] Using grid-marginalized ICAR with per-dimension length scales")
        lnsigma_range = tuple(pixelpop_data.coupling_prior[0])
        ICAR_model = None  # constructed directly in nonparametric_model
        grid_icar = grid_marginalized_ICAR_length_scales(
            single_dimension_adj_matrices=pixelpop_data.adj_matrices,
            lnsigma_ranges=lnsigma_range,
            grid_points=100,
            is_sparse=True,
        )
    elif pixelpop_data.marginalize_sigma:
        ICAR_model = sigma_marginalized_ICAR
    else:
        ICAR_model = ICAR_length_scales

    def nonparametric_model(event_bins, inj_bins, event_weights, inj_weights, skip=False):
        """
        Evaluate the nonparametric (ICAR/CAR) pixelized model contribution.

        Either samples the log merger rate density from an intrinsic CAR prior
        (with optional length scales) or falls back to a log-rate-only model if
        skipped.

        Parameters
        ----------
        event_bins : ndarray
            Indices mapping events into multidimensional bins.
        inj_bins : ndarray
            Indices mapping injections into multidimensional bins.
        event_weights : ndarray
            Current accumulated event log-weights.
        inj_weights : ndarray
            Current accumulated injection log-weights.
        skip : bool, optional
            If True, skip the ICAR model and use only a single log-rate parameter.

        Returns
        -------
        event_weights : ndarray
            Updated event log-weights including nonparametric contributions.
        inj_weights : ndarray
            Updated injection log-weights including nonparametric contributions.
        """

        if skip:
            R = numpyro.sample('log_rate', dist.ImproperUniform(dist.constraints.real, (), ()))
            return event_weights + R[None,None], inj_weights + R[None]

        if not pixelpop_data.marginalize_sigma:
            coupling_prior = pixelpop_data.coupling_prior
            if pixelpop_data.length_scales:
                lsigma = numpyro.sample('lnsigma', coupling_prior[1](*coupling_prior[0]), sample_shape=(pixelpop_data.dimension,))
            else:
                lsigma = numpyro.sample('lnsigma', coupling_prior[1](*coupling_prior[0]), sample_shape=())

        if pixelpop_data.diagonalize_icar:
            # Sample in the Gaussian IID ("eigenbasis") space and map to the
            # merger rate density with DiagonalizedICARTransform. This mirrors the
            # experimental prior_probabilistic_model and tends to give NUTS a nicer
            # (closer-to-isotropic) geometry than sampling merger_rate_density
            # directly from the ICAR distribution. lsigma was drawn above.
            _eigenbasis_sites = numpyro.sample(
                '_eigenbasis_sites',
                dist.Normal(0., 1.).expand(pixelpop_data.bins),
            )
            # Pin the zero-mode (the constant/DC eigenvector) to 0: the transform
            # would otherwise scale it by the regularized eigenvalue, imposing a
            # spurious proper prior on the overall offset. With it pinned, the
            # eigenbasis carries only shape, and the overall rate is restored by a
            # free improper-uniform log-rate offset added below.
            eigenbasis_sites = _eigenbasis_sites.at[(0,) * pixelpop_data.dimension].set(0.)
            if pixelpop_data.spde_wkb:
                # WKB field: each (log) hyperparameter varies linearly, intercept + slope . x.
                dim = pixelpop_data.dimension
                X = pixelpop_data.spde_coords                      # (*bins, dim)
                # Intercepts (log-sigma is lsigma, sampled above) and slopes.
                l_0 = numpyro.sample('log_ranges', pixelpop_data.range_prior[1](*pixelpop_data.range_prior[0]), sample_shape=(dim,))
                log_nu_0 = numpyro.sample('log_nu_spde', pixelpop_data.smoothness_prior[1](*pixelpop_data.smoothness_prior[0]))
                A_range = numpyro.sample('range_response', pixelpop_data.range_response_prior[1](*pixelpop_data.range_response_prior[0]), sample_shape=(dim, dim))
                a_nu = numpyro.sample('nu_slope', pixelpop_data.nu_slope_prior[1](*pixelpop_data.nu_slope_prior[0]), sample_shape=(dim,))
                a_sigma = numpyro.sample('sigma_slope', pixelpop_data.sigma_slope_prior[1](*pixelpop_data.sigma_slope_prior[0]), sample_shape=(dim,))
                # Build the linear-in-log spatial fields.
                log_ranges_field = l_0.reshape((dim,) + (1,) * dim) + jnp.einsum('ij,...j->i...', A_range, X)
                nu_field = jnp.exp(log_nu_0 + jnp.einsum('i,...i->...', a_nu, X))
                log_sigma_field = lsigma + jnp.einsum('i,...i->...', a_sigma, X)
                transform = WKBNonStationaryMaternSPDETransform(
                    log_sigma_field, log_ranges_field, nu_field, pixelpop_data.adj_matrices, is_sparse=True)
            elif pixelpop_data.spde_matern:
                nu_spde = jnp.exp(numpyro.sample('log_nu_spde', pixelpop_data.smoothness_prior[1](*pixelpop_data.smoothness_prior[0])))
                log_ranges = numpyro.sample('log_ranges', pixelpop_data.range_prior[1](*pixelpop_data.range_prior[0]), sample_shape=(pixelpop_data.dimension,))
                transform = MaternSPDETransform(lsigma, log_ranges, nu_spde, pixelpop_data.adj_matrices, is_sparse=True)
            else:
                transform = DiagonalizedICARTransform(lsigma, pixelpop_data.adj_matrices, is_sparse=True)
            transformed = transform(eigenbasis_sites)
            if pixelpop_data.lower_triangular:
                # Symmetrize, equivalent to sampling from the symmetrized space.
                transformed = 0.5 * (transformed + transformed.swapaxes(0, 1))
            # Free overall log-rate offset (flat prior), carrying the absolute-rate
            # information that the pinned zero-mode removed. The 'log_rate'
            # deterministic computed below picks this up via LSE(field + c) = LSE(field) + c.
            log_rate_offset = numpyro.sample(
                'log_rate_offset', dist.ImproperUniform(dist.constraints.real, (), ())
                )
            transformed = transformed + log_rate_offset
            merger_rate_density = numpyro.deterministic('merger_rate_density', transformed)

        elif pixelpop_data.cauchy_icar:
            if not pixelpop_data.lower_triangular:
                merger_rate_density = numpyro.sample(
                        'merger_rate_density',
                        ICAR_model(
                            nu=1., 
                            log_sigmas=lsigma,
                            single_dimension_adj_matrices=pixelpop_data.adj_matrices,
                            dof_correction=1.,
                            is_sparse=True,
                        ),
                    )
            else:
                base_interpolation = numpyro.sample('base_interpolation', dist.ImproperUniform(dist.constraints.real, unique_sample_shape, ()))
                merger_rate_density = numpyro.deterministic('merger_rate_density', lt_map(base_interpolation))
                
                prior_factor = ICAR_model(
                            nu=1., 
                            log_sigmas=lsigma,
                            single_dimension_adj_matrices=pixelpop_data.adj_matrices,
                            dof_correction=normalization_dof / np.prod(pixelpop_data.bins),
                            is_sparse=True,
                        ).log_prob(merger_rate_density)
                
                numpyro.factor('prior_factor', prior_factor)
            numpyro.factor('tail_regularization', -jnp.sum(((merger_rate_density - jnp.mean(merger_rate_density)) / 100)**2 / 2))
        
        elif pixelpop_data.lower_triangular:
            
            base_interpolation = numpyro.sample('base_interpolation', dist.ImproperUniform(dist.constraints.real, unique_sample_shape, ()))
            merger_rate_density = numpyro.deterministic('merger_rate_density', lt_map(base_interpolation))
            
            if pixelpop_data.marginalize_sigma:
                prior_factor, quad = lower_triangular_sigma_marg_log_prob_and_log_quad(merger_rate_density, normalization_dof, pixelpop_data.adj_matrices)
            else:
                prior_factor = lower_triangular_log_prob(merger_rate_density, normalization_dof, lsigma, pixelpop_data.adj_matrices)
            numpyro.factor('prior_factor', prior_factor)

        else:

            if pixelpop_data.marginalize_sigma and pixelpop_data.length_scales:
                merger_rate_density = numpyro.sample(
                    'merger_rate_density',
                    dist.ImproperUniform(
                        dist.constraints.real,
                        tuple(pixelpop_data.bins),
                        (),
                    ),
                )
                prior_factor = grid_icar.log_prob(merger_rate_density)
                numpyro.factor('prior_factor', prior_factor)
            elif pixelpop_data.marginalize_sigma:
                icar = ICAR_model(single_dimension_adj_matrices=pixelpop_data.adj_matrices, is_sparse=True)
                merger_rate_density = numpyro.sample(
                    'merger_rate_density',
                    dist.ImproperUniform(
                        dist.constraints.real,
                        tuple(pixelpop_data.bins),
                        (),
                    ),
                )
                prior_factor, quad = icar.log_prob_and_quad(merger_rate_density)
                numpyro.factor('prior_factor', prior_factor)
            else:
                merger_rate_density = numpyro.sample(
                    'merger_rate_density',
                    ICAR_model(
                        log_sigmas=lsigma,
                        single_dimension_adj_matrices=pixelpop_data.adj_matrices,
                        is_sparse=True,
                    ),
                )

        if not pixelpop_data.lower_triangular:
            normalization = numpyro.deterministic('log_rate', LSE(merger_rate_density)+jnp.sum(pixelpop_data.logdV))
            for ii, p in enumerate(pixelpop_data.pixelpop_parameters):
                sum_axes = tuple(np.arange(pixelpop_data.dimension)[np.r_[0:ii,ii+1:pixelpop_data.dimension]])
                numpyro.deterministic(
                    f'log_marginal_{p}',
                    LSE(merger_rate_density-normalization, axis=sum_axes)
                    + jnp.sum(pixelpop_data.logdV[:ii])
                    + jnp.sum(pixelpop_data.logdV[ii+1:])
                )

        if pixelpop_data.marginalize_sigma and pixelpop_data.length_scales:
            # Compute conditional moments of p(lnsigma | phi) on the grid.
            _, cond_log_weights = grid_icar.log_prob_and_conditional_lnsigma(merger_rate_density)
            flat_log_weights = cond_log_weights.ravel()
            weights = jnp.exp(flat_log_weights - LSE(flat_log_weights))  # softmax
            meshes = jnp.meshgrid(*grid_icar._grids_1d, indexing='ij')
            lnsigma_grid_flat = jnp.stack(meshes, axis=-1).reshape(-1, pixelpop_data.dimension)
            # Conditional mean and std: shape (D,)
            cond_mean = jnp.sum(weights[:, None] * lnsigma_grid_flat, axis=0)
            cond_std = jnp.sqrt(jnp.sum(weights[:, None] * (lnsigma_grid_flat - cond_mean) ** 2, axis=0))
            # Sample lnsigma from the Gaussian approximation of p(lnsigma | phi).
            numpyro.sample('lnsigma', dist.Normal(cond_mean, cond_std).to_event(1))
        elif pixelpop_data.marginalize_sigma:
            unscaled_gamma = numpyro.sample('unscaled_gamma', numpyro.distributions.Gamma(concentration=(normalization_dof/2)))
            precision = 2 * unscaled_gamma / quad
            numpyro.deterministic('lnsigma', -0.5*jnp.log(precision))

        # Use the raw ICAR field for event and injection weights; any
        # parametric windows are applied in parametric_model.
        if pixelpop_data.IID:
            event_weights += merger_rate_density[event_bins[0]]
            inj_weights += merger_rate_density[inj_bins[0]]

            event_weights += merger_rate_density[event_bins[1]] - normalization
            inj_weights += merger_rate_density[inj_bins[1]] - normalization
        else:
            event_weights += merger_rate_density[event_bins]
            inj_weights += merger_rate_density[inj_bins]
        if log == 'debug':
            jaxprint('[DEBUG] pixelpop LSE(event_weights)={ew}, LSE(injection_weights)={iw}', ew=LSE(event_weights), iw=LSE(inj_weights))
        return event_weights, inj_weights

    def probabilistic_model(posteriors, injections):
        """
        Full probabilistic model for hierarchical GW population inference.

        Combines the nonparametric pixelized rate density with parametric models,
        applies detection efficiency corrections, and evaluates the likelihood.

        Parameters
        ----------
        posteriors : dict
            Posterior samples from detected events.
        injections : dict
            Injection data including selection effects.

        Side Effects
        ------------
        Stores deterministic nodes in NumPyro for logging:
        - log_likelihood
        - log_likelihood_variance
        - pe_variance
        - vt_variance
        - Nexp

        Returns
        -------
        None
            (Factors likelihood into NumPyro’s computation graph.)
        """
        if pixelpop_data.IID:
            eb = [pixelpop_data.event_bins_1, pixelpop_data.event_bins_2]
            ib = [pixelpop_data.inj_bins_1, pixelpop_data.inj_bins_2]
        else:
            eb, ib = pixelpop_data.event_bins, pixelpop_data.inj_bins

        event_weights, inj_weights = nonparametric_model(
            eb, ib,
            posteriors['ln_dVTc']-posteriors['log_prior'],
            injections['ln_dVTc']-injections['log_prior'],
            skip=pixelpop_data.skip_nonparametric
            )
        event_weights, inj_weights = parametric_model(
            posteriors, 
            injections, 
            event_weights, 
            inj_weights
            )

        likelihood_dict = \
            rate_likelihood(
                event_weights,
                inj_weights,
                injections['total_generated'],
                live_time=injections['analysis_time'],
                event_counts=pixelpop_data.event_counts
                )
        
        ln_likelihood  =likelihood_dict['log_likelihood']
        nexp = likelihood_dict['nexp']
        pe_var = likelihood_dict['total_pe_lnL_variance']
        vt_var = likelihood_dict['total_vt_lnL_variance']
        total_var = likelihood_dict['total_lnL_variance']        
        
        taper = smooth(total_var, pixelpop_data.UncertaintyCut**2, 0.1) # "smooth" cutoff above Talbot+Golomb 2022 recommendation to retain autodifferentiability
        
        if pixelpop_data.EventNeffCut > 0.:
            numpyro.factor("single_event_neff_taper", jnp.sum(smooth(
                -jnp.log(likelihood_dict['single_event_neffs']),
                -jnp.log(pixelpop_data.EventNeffCut),
                0.1))
            )
        if pixelpop_data.SelectionNeffCut:
            numpyro.factor("selection_neff_taper", smooth(
                -jnp.log(likelihood_dict['vt_neff']),
                -jnp.log(4*pixelpop_data.Nobs),
                0.1
                ))
        # save these values!
        numpyro.deterministic("log_likelihood", ln_likelihood)
        numpyro.deterministic("log_likelihood_variance", total_var)
        numpyro.deterministic("pe_variance", pe_var)
        numpyro.deterministic("vt_variance", vt_var)
        numpyro.deterministic("Nexp", nexp)

        numpyro.factor("log_likelihood_plus_taper", ln_likelihood + taper)

    return probabilistic_model, initial_value

#: Sites this size or smaller are diagnosed in one call rather than sliced.
BULK_DIAGNOSTIC_PARAMETERS = 4096


def _arg_best(values, argbest):
    """Pick out the best of `values` and the index tuple that located it."""
    values = np.asarray(values)
    flat = int(argbest(values))
    index = np.unravel_index(flat, values.shape) if values.ndim else ()
    return float(values.reshape(-1)[flat]), tuple(int(p) for p in index)


def _worst_over_site(value):
    """
    Largest split R-hat and smallest Neff within one site, with their indices.

    ``value`` is ``(num_draws, *event_shape)``. Big sites are walked a slice of the
    leading event axis at a time: same arithmetic, but the FFT inside
    ``effective_sample_size`` allocates per slice rather than per site.
    """
    if value.ndim == 1 or value[0].size <= BULK_DIAGNOSTIC_PARAMETERS:
        return (_arg_best(split_gelman_rubin(value[None, ...]), np.argmax),
                _arg_best(effective_sample_size(value[None, ...]), np.argmin))

    worst_rhat, worst_neff = (-np.inf, ()), (np.inf, ())
    for i in range(value.shape[1]):
        block = value[:, i][None, ...]
        rhat, rhat_index = _arg_best(split_gelman_rubin(block), np.argmax)
        neff, neff_index = _arg_best(effective_sample_size(block), np.argmin)
        if rhat > worst_rhat[0]:
            worst_rhat = (rhat, (i, *rhat_index))
        if neff < worst_neff[0]:
            worst_neff = (neff, (i, *neff_index))
    return worst_rhat, worst_neff


def get_worst_rhat_neff(chain_samples, skip_keys=[], latent_sites=None):
    """
    Identify the parameter with the worst R-hat and effective sample size (Neff).

    Parameters
    ----------
    chain_samples : dict
        Dictionary of chain samples from NumPyro MCMC, with parameter name keys
    skip_keys : list
        Names of sites to leave out of the search.
    latent_sites : set of str, optional
        Restrict the search to these sites, as returned by :func:`get_latent_sites`.
        Deterministic sites are functions of the latents, so their diagnostics are
        redundant and the big ones dominate the cost. Defaults to every key.

    Returns
    -------
    rhat_key : str
        Name of parameter with the largest R-hat.
    rhat_chain : ndarray
        Sample chain of the worst R-hat parameter.
    neff_key : str
        Name of parameter with the smallest Neff.
    neff_chain : ndarray
        Sample chain of the worst Neff parameter.
    """
    keys = [k for k in chain_samples
            if k not in skip_keys and (latent_sites is None or k in latent_sites)]
    if not keys:  # nothing survived the filters
        keys = [k for k in chain_samples if k not in skip_keys]

    # Seeded from the first site rather than +-inf: a site whose diagnostics come
    # back NaN loses every comparison, and with no seed that would leave nothing to
    # report at all.
    worst_rhat = worst_neff = None
    for key in keys:
        (rhat, rhat_index), (neff, neff_index) = _worst_over_site(np.asarray(chain_samples[key]))
        if worst_rhat is None or rhat > worst_rhat[0]:
            worst_rhat = (rhat, key, rhat_index)
        if worst_neff is None or neff < worst_neff[0]:
            worst_neff = (neff, key, neff_index)

    def label_and_chain(worst):
        _, key, index = worst
        return (f'{key}{[int(p) for p in index]}'.replace('[]', ''),
                chain_samples[key][(..., *index)])

    rhat_key, rhat_chain = label_and_chain(worst_rhat)
    neff_key, neff_chain = label_and_chain(worst_neff)
    return rhat_key, rhat_chain, neff_key, neff_chain

def trace_model(probabilistic_model, initial_value={}, model_kwargs={}):
    """
    Trace the probabilistic model once at ``initial_value``.

    Parameters
    ----------
    probabilistic_model : callable
        NumPyro probabilistic model.
    initial_value : dict, optional
        Dictionary of initial parameter values to condition on.
    model_kwargs : dict, optional
        Keyword arguments for the probabilistic model (e.g., posterior and injection data).

    Returns
    -------
    trace : dict
        NumPyro trace, keyed by site name.
    """
    conditioned_model = handlers.condition(probabilistic_model, data=initial_value)
    with handlers.seed(rng_seed=0):
        return handlers.trace(conditioned_model).get_trace(**model_kwargs)

def get_latent_sites(trace, initial_value={}):
    """
    Names of the sites NUTS will actually sample.

    Parameters
    ----------
    trace : dict
        NumPyro trace, as returned by :func:`trace_model`.
    initial_value : dict, optional
        The values the trace was conditioned on.

    Returns
    -------
    latent_sites : set of str
        Names of the unobserved sample sites.
    """
    # Conditioning marks the initialized sites observed even though they are latent
    # in the unconditioned model, so they are added back by name. numpyro.factor
    # sites are genuinely observed and stay out, as do deterministic sites.
    return {
        name for name, site in trace.items()
        if site['type'] == 'sample'
        and (name in initial_value or not site.get('is_observed', False))
    }

def get_table_size(probabilistic_model, initial_value, model_kwargs, print_keys, trace=None):
    """
    Calculate the size of the in-progress summary table.

    Parameters
    ----------
    probabilistic_model : callable
        NumPyro probabilistic model.
    initial_value : dict
        Dictionary of initial parameter values.
    model_kwargs : dict
        Keyword arguments for the probabilistic model (e.g., posterior and injection data).
    print_keys : list of str
        Keys for which to include values in the summary table.
    trace : dict, optional
        Pre-computed trace of the model, to avoid tracing it twice.

    Returns
    -------
    size : int
        Number of rows expected in the summary table.
    """
    if trace is None:
        trace = trace_model(probabilistic_model, initial_value, model_kwargs)

    size = 2
    for name in print_keys:
        if name.startswith('~'):
            continue
        try:
            size += trace[name]["value"].size
        except KeyError:
            raise KeyError(f'You are trying to print \"{name}\", valid print_keys are {list(trace.keys())}')
    return size

PARAMETRIC_DENSE_MASS = 'parametric'

def parametric_dense_blocks(pixelpop_data):
    """
    One dense mass-matrix block per parametric model dimension.

    Each block holds the hyperparameters of a single model, so NUTS learns the
    within-model correlations (e.g. the power-law slopes against the peak
    locations) without paying for a full dense matrix over every site.

    Parameters
    ----------
    pixelpop_data : PixelPopData
        Run configuration, read for ``other_parameters`` and
        ``parameter_to_hyperparameters``.

    Returns
    -------
    blocks : list of tuple of str
        Hyperparameter names, grouped by the model that consumes them.
    """
    return [
        tuple(pixelpop_data.parameter_to_hyperparameters[p])
        for p in pixelpop_data.other_parameters
    ]

def resolve_dense_mass(dense_mass, pixelpop_data=None, latent_sites=None):
    """
    Expand a dense mass-matrix specification into numpyro's list-of-tuples form.

    Accepts, in addition to numpyro's own ``True``/``False``/list-of-tuples:

    - ``'parametric'``, which expands to one block per parametric model, i.e.
      ``[('alpha_1', 'alpha_2', ..., 'mpp_1', ...), ('lamb', 'max_z'), ...]``;
    - blocks that name *model dimensions* rather than hyperparameters, so
      ``[('log_mass_1', 'mass_ratio', 'redshift'), ('a', 't')]`` correlates the
      masses with the redshift and, separately, the spin magnitudes with the
      tilts. Hyperparameter and model-dimension names may be mixed freely in one
      block;
    - ``'parametric'`` as an element, which expands in place, so
      ``['parametric', ('log_nu_spde', 'log_ranges', 'lnsigma')]`` blocks the
      parametric models and the nonparametric field separately.

    Names that are not sampled -- Delta priors, or sites that this run's model
    does not have -- are dropped rather than raising, and a name repeated across
    blocks stays only in the first, since numpyro requires the blocks to
    partition the latent sites.

    Parameters
    ----------
    dense_mass : bool, str, or sequence
        The specification to expand.
    pixelpop_data : PixelPopData, optional
        Run configuration. Required to expand ``'parametric'`` or model-dimension
        names.
    latent_sites : set of str, optional
        Names of the sites NUTS samples, from :func:`get_latent_sites`. Anything
        outside this set is dropped. If omitted, only Delta-prior hyperparameters
        are dropped.

    Returns
    -------
    dense_mass : bool or list of tuple of str
        Ready to hand to ``numpyro.infer.NUTS``. ``False`` if no block survived.
    """
    if dense_mass is None:
        return False
    if isinstance(dense_mass, bool):
        return dense_mass
    if isinstance(dense_mass, str):
        dense_mass = [dense_mass]

    if pixelpop_data is None:
        model_dimensions, parameter_hypers, priors = set(), {}, {}
    else:
        model_dimensions = set(pixelpop_data.other_parameters)
        parameter_hypers = pixelpop_data.parameter_to_hyperparameters
        priors = pixelpop_data.priors
    # A reparameterized hyperparameter is a deterministic, not a latent site, so a
    # block naming it would otherwise be silently dropped along with the
    # correlations it was asked to capture.
    reparameterized = reparameterized_sites(priors)

    groups = []
    for group in dense_mass:
        if group == PARAMETRIC_DENSE_MASS:
            if pixelpop_data is None:
                raise ValueError(
                    f"dense_mass='{PARAMETRIC_DENSE_MASS}' needs pixelpop_data to "
                    "know which hyperparameters belong to which model"
                )
            groups.extend(parametric_dense_blocks(pixelpop_data))
        elif isinstance(group, str):
            groups.append((group,))
        else:
            groups.append(tuple(group))

    blocks, seen = [], set()
    for group in groups:
        block = []
        for name in group:
            names = parameter_hypers[name] if name in model_dimensions else [name]
            for hyper in names:
                site = reparameterized.get(hyper, hyper)
                if site in seen:
                    continue
                if latent_sites is not None and site not in latent_sites:
                    continue
                if hyper in priors and priors[hyper][1].__name__ == 'Delta':
                    continue
                seen.add(site)
                block.append(site)
        if block:
            blocks.append(tuple(block))
    return blocks or False

def inference_loop(
    probabilistic_model, model_kwargs={}, initial_value={}, warmup=10000, tot_samples=100, thinning=100, pacc=0.65, maxtreedepth=10,
    num_samples=1, parallel=1, rng_key=random.PRNGKey(1), cache_cadence=1, run_dir='./', name='',
    print_keys=['Nexp', 'log_likelihood', 'log_likelihood_variance'], dense_mass=False, chain_offset=0,
    pixelpop_data=None
    ):
    """
    Run MCMC inference with a probabilistic model and return posterior samples.

    This function manages warmup, thinning, caching, diagnostics, and saving of
    posterior samples across multiple independent chains.

    Parameters
    ----------
    probabilistic_model : callable
        NumPyro probabilistic model to sample from.
    model_kwargs : dict, optional
        Arguments passed to the probabilistic model (e.g., posterior and injection data).
    initial_value : dict, optional
        Initial parameter values for warmup.
    warmup : int, optional
        Number of warmup iterations (default 10000).
    tot_samples : int, optional
        Total number of posterior samples to save per chain.
    thinning : int, optional
        Interval between recorded samples (default 100).
    pacc : float, optional
        Target acceptance probability for NUTS (default 0.65).
    maxtreedepth : int, optional
        Maximum tree depth for NUTS (default 10).
    num_samples : int, optional
        Frequency of printing chain diagnostics (default 1).
    parallel : int or list, optional
        Number of independent chains to run (default 1). If list of ints, 
        specifies number in name of savefile (chain_{num}_samples.h5).
    rng_key : jax.random.PRNGKey, optional
        Random key for reproducibility.
    cache_cadence : int, optional
        Interval (in samples) between checkpoint saves (default 1).
    run_dir : str, optional
        Directory to save output chains (default "./").
    name : str, optional
        Subdirectory name for this run.
    print_keys : list of str, optional
        Keys to include in periodic summaries (default ["Nexp", "log_likelihood", "log_likelihood_variance"]).
    dense_mass : bool, str, or sequence, optional
        Dense mass-matrix specification (default False), expanded by
        :func:`resolve_dense_mass`. Beyond numpyro's own ``True``/``False``/
        list-of-tuples this takes ``'parametric'`` (one dense block per parametric
        model) and blocks named by model dimension rather than hyperparameter.
    chain_offset : int, optional
        Offset applied to chain index when saving outputs (default 0).
    pixelpop_data : PixelPopData, optional
        Run configuration, needed only to expand a ``dense_mass`` specification
        that names models rather than hyperparameters.

    Returns
    -------
    samples : list of dict
        List of posterior samples for each chain.
    mcmc : numpyro.infer.MCMC
        Completed MCMC sampler instance.
    """

    trace = trace_model(probabilistic_model, initial_value, model_kwargs)
    table_size = get_table_size(probabilistic_model, initial_value, model_kwargs, print_keys, trace=trace)
    latent_sites = get_latent_sites(trace, initial_value)
    dense_mass = resolve_dense_mass(dense_mass, pixelpop_data, latent_sites=latent_sites)
    if isinstance(dense_mass, list):
        print(f"Dense mass-matrix blocks: {dense_mass}")
    skip_keys = [k[1:] for k in print_keys if k.startswith('~')]

    # Bind the data in rather than passing it to mcmc.run(): numpyro keys its compiled-step cache
    # on the model kwargs, and our nested dicts are unhashable, so it recompiles every restart.
    # Caching only -- the data is a compile-time constant either way, so samples are unchanged.
    bound_model = partial(probabilistic_model, **model_kwargs) if model_kwargs else probabilistic_model

    kernel = NUTS(bound_model, max_tree_depth=maxtreedepth, target_accept_prob=pacc, init_strategy=numpyro.infer.init_to_value(values=initial_value), dense_mass=dense_mass)

    samples = []
    if not isinstance(parallel, (list, tuple)):
        parallel = list(range(parallel))
    rng_keys = random.split(rng_key, num=len(parallel))
    for ii, chain in enumerate(parallel):
        rng_key = rng_keys[ii]
        print(f"Warming up chain #{chain} out of {parallel}")
        # NOTE: leave progress_bar at its default; progress_bar=False recompiles every restart.
        mcmc = MCMC(kernel, thinning=thinning, num_warmup=warmup, num_samples=num_samples*thinning, num_chains=1)# , chain_method='vectorized')# , chain_method='sequential') # vectorized is an experimental method. We can pass 'parallel' which attempts to distribute the chains across multiple GPUs, e.g. on pcdev12 we could do num_chains = 4 across the a100s. If num_chains is too large, it defaults to 'sequential' which simply evaluates the chains in series.
        
        mcmc.warmup(rng_key)
        sys.stdout.write("\n"*(table_size+3)) # buffer line between the progress bars
        chain_samples, filled = None, 0
        mcmc.transfer_states_to_host()
        num_chunks = int(1e-4 + tot_samples/num_samples)
        sample_iterator = tqdm(range(num_chunks))
        sample_iterator.set_description("drawing thinned samples")
        for sample in sample_iterator:
            mcmc.post_warmup_state = mcmc.last_state
            mcmc.run(mcmc.post_warmup_state.rng_key)
            next_sample = mcmc.get_samples()
            sys.stdout.write("\x1b[1A\n\x1b[1A")

            # Allocate the whole chain once and fill slices. Growing by concatenation
            # instead holds the old and the new array at the same time, which on the
            # last chunks of a 3D run is several GB of transient.
            chunk = len(next_sample[next(iter(next_sample))])
            if chain_samples is None:
                chain_samples = {
                    key: np.empty((num_chunks*chunk, *value.shape[1:]), dtype=value.dtype)
                    for key, value in next_sample.items()
                    }
            for key, value in chain_samples.items():
                value[filled:filled+chunk] = next_sample[key]
            filled += chunk
            # views onto the filled part, so nothing downstream sees the unwritten tail
            collected = {key: value[:filled] for key, value in chain_samples.items()}

            mcmc.transfer_states_to_host()
            # always save the last chunk, so the h5 holds the whole chain even when
            # cache_cadence does not divide the number of chunks
            last_chunk = sample == num_chunks - 1
            if (sample % cache_cadence == 0 or last_chunk) and (filled >= 4):
                sys.stdout.write(f"\x1b[1A\x1b[2K"*(table_size+3)) # move the cursor up to overwrite the summary table for the NEXT print

                rhat, rhat_chain, neff, neff_chain = get_worst_rhat_neff(
                    collected, skip_keys=skip_keys, latent_sites=latent_sites)
                summary_dict = {key: collected[key] for key in print_keys if key[1:] not in skip_keys}
                summary_dict['worst r_hat: '+rhat] = rhat_chain
                summary_dict['worst n_eff: '+neff] = neff_chain

                print_summary(summary_dict, group_by_chain=False)
                os.makedirs(os.path.join(run_dir, name), exist_ok=True)
                with open(os.path.join(run_dir, name, f'chain_{chain+chain_offset}_metadata.txt'), 'w+') as f:
                    with redirect_stdout(f):
                        print_summary(summary_dict, group_by_chain=False)
                f = os.path.join(run_dir, name, f'chain_{chain+chain_offset}_samples.h5')
                h5ify.save(f, collected, mode='w')

        samples.append({key: value[:filled] for key, value in chain_samples.items()})

    return samples, mcmc