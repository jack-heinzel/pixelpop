import numpy as np
from ..utils.nearest_neighbor import create_CAR_coupling_matrix
from ..models.gwpop_models import * 
from ..models.car import lower_triangular_map, mult_outer, add_outer
from .car import (DiagonalizedICARTransform, initialize_ICAR_normalized, initialize_sigma_marginalized_ICAR, initialize_ICAR_normalized_lower_triangular)
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.debug import print as jaxprint
from ..utils.data import place_in_bins
from jax.scipy.special import logsumexp as LSE
import numpyro

def init_interpolant_from_counts(pixelpop_data, alpha=1.0):
    """
    Build a data-informed init from event bin occupancy.

    alpha: pseudocount to avoid log(0)
    """
    bins = tuple(int(b) for b in pixelpop_data.bins)
    n_total = int(np.prod(bins))
    log_dx = jnp.sum(pixelpop_data.logdV)  # scalar log(cell volume)

    # event_bins is typically a tuple/list length=dimension,
    # each entry shaped (...), e.g. (Nevents, Nsamps) or (Ntotal,)
    event_bins = pixelpop_data.event_bins

    # Convert multi-index -> linear index, then FLATTEN to 1D
    lin = jnp.ravel_multi_index(event_bins, dims=bins)
    lin = jnp.asarray(lin).reshape(-1)
    lin = lin.astype(jnp.int32)

    # bincount needs 1D
    counts = jnp.bincount(lin, length=n_total).reshape(bins)

    # log of (counts + pseudocount)
    logp = jnp.log(counts + alpha)

    # Normalize to a log-DENSITY: enforce LSE(logp) = -log_dx
    init = logp - LSE(logp) - log_dx
    return init


def init_interpolant_from_counts_lt(pixelpop_data, alpha=1.0):
    """
    Build a data-informed init for the lower-triangular normalized PixelPop.

    Counts are accumulated on the full bins-by-bins grid, folded across the
    diagonal of the first two dimensions (so a pair contributes to the same
    LT cell regardless of ordering), then the unique LT entries are
    extracted and renormalized so LSE(init_lt) + sum(logdV) = 0.
    """
    bins = tuple(int(b) for b in pixelpop_data.bins)
    bins_first = bins[0]
    n_total = int(np.prod(bins))
    log_dx = jnp.sum(pixelpop_data.logdV)

    event_bins = pixelpop_data.event_bins
    lin = jnp.ravel_multi_index(event_bins, dims=bins)
    lin = jnp.asarray(lin).reshape(-1).astype(jnp.int32)
    counts = jnp.bincount(lin, length=n_total).reshape(bins)

    # Fold counts across the (0,1)-diagonal so both orderings hit the same LT cell.
    counts_sym = counts + jnp.swapaxes(counts, 0, 1)
    diag_idx = jnp.arange(bins_first)
    counts_sym = counts_sym.at[diag_idx, diag_idx].set(counts[diag_idx, diag_idx])

    # triu_indices ordering matches lower_triangular_map storage convention.
    iu_a, iu_b = jnp.triu_indices(bins_first)
    counts_lt = counts_sym[iu_a, iu_b]  # shape (tri_size, *bins[2:])

    logp = jnp.log(counts_lt + alpha)
    init = logp - LSE(logp) - log_dx
    return init


def setup_mixture_probabilistic_model_direct(pixelpop_data, log="default"):
    """
    Construct a mixture probabilistic model for GW population inference.

    Returns
    -------
    probabilistic_model : callable
    initial_value : dict
    """

    bins = tuple(int(b) for b in pixelpop_data.bins)
    if pixelpop_data.lower_triangular:
        bins_first = bins[0]
        assert bins[1] == bins_first, (
            "lower_triangular requires the first two dimensions to share size; "
            f"got bins[0]={bins_first}, bins[1]={bins[1]}"
        )
        tri_size = bins_first * (bins_first + 1) // 2
        unique_sample_shape = (tri_size,) + tuple(bins[2:])
        normalization_dof = int(np.prod(unique_sample_shape))
        lt_map_fn = lower_triangular_map(bins_first)
        ICAR_model = initialize_ICAR_normalized_lower_triangular(
            pixelpop_data.dimension, bins_first
        )
    else:
        unique_sample_shape = bins
        normalization_dof = int(np.prod(bins))
        lt_map_fn = None
        ICAR_model = initialize_ICAR_normalized(pixelpop_data.dimension)

    # ----------------------------------------------------------------
    # Initial values
    # ----------------------------------------------------------------
    if pixelpop_data.skip_nonparametric:
        initial_value = {
            "log_rate": jnp.log(50.),
        }
    else:
        if pixelpop_data.lower_triangular:
            init_interpolant = init_interpolant_from_counts_lt(pixelpop_data)
        else:
            init_interpolant = init_interpolant_from_counts(pixelpop_data)

        initial_value = {
            "interpolant": init_interpolant,
            "log_rate": jnp.log(50.),
        }

    # ----------------------------------------------------------------
    # Parametric model: split into "common" and "strong"
    # ----------------------------------------------------------------
    def parametric_model(data, injections):
        sample = {}

        common_param_event_weights = 0
        common_param_inj_weights = 0
        strong_param_event_weights = 0
        strong_param_inj_weights = 0

        # Draw all hyperparameters
        for key in pixelpop_data.priors:
            args, distribution = pixelpop_data.priors[key]
            if distribution.__name__ == "Delta":
                sample[key] = args[0]
            else:
                sample[key] = numpyro.sample(key, distribution(*args))

        # Draw mixture-specific hyperparameters (not already drawn)
        for key in pixelpop_data.mixture_priors:
            if key not in sample:
                args, distribution = pixelpop_data.mixture_priors[key]
                if distribution.__name__ == "Delta":
                    sample[key] = args[0]
                else:
                    sample[key] = numpyro.sample(key, distribution(*args))

        if log == "debug":
            for p in (pixelpop_data.other_parameters
                      + pixelpop_data.pixelpop_parameters):
                jaxprint("[DEBUG] =================================")
                jaxprint("[DEBUG] parametric parameters: {p}", p=p)
                for k in pixelpop_data.parameter_to_hyperparameters.get(p, []):
                    if k in sample:
                        jaxprint("[DEBUG]   {k} = {s}", k=k, s=sample[k])

        for constraint_func in pixelpop_data.constraint_funcs:
            numpyro.factor(constraint_func.__name__, constraint_func(sample))

        # Common parameters (multiply entire mixture)
        for p in pixelpop_data.other_parameters:
            model_fn = pixelpop_data.parametric_models[p]
            hypers = [
                sample[h]
                for h in pixelpop_data.parameter_to_hyperparameters[p]
            ]
            common_param_event_weights += model_fn(data, *hypers)
            common_param_inj_weights += model_fn(injections, *hypers)

            if log == "debug":
                jaxprint(
                    "[DEBUG] common {p}: LSE(ew)={ew}, LSE(iw)={iw}",
                    p=p,
                    ew=LSE(common_param_event_weights),
                    iw=LSE(common_param_inj_weights),
                )

        # Strong parameters (parametric side of mixture only)
        for p in pixelpop_data.pixelpop_parameters:
            model_fn = pixelpop_data.mixture_parametric_models[p]
            hypers = [
                sample[h]
                for h in pixelpop_data.mixture_parameter_to_hyperparameters[p]
            ]
            strong_param_event_weights += model_fn(data, *hypers)
            strong_param_inj_weights += model_fn(injections, *hypers)

            if log == "debug":
                jaxprint(
                    "[DEBUG] strong {p}: LSE(ew)={ew}, LSE(iw)={iw}",
                    p=p,
                    ew=LSE(strong_param_event_weights),
                    iw=LSE(strong_param_inj_weights),
                )

        return (
            common_param_event_weights,
            common_param_inj_weights,
            strong_param_event_weights,
            strong_param_inj_weights,
        )

    # ----------------------------------------------------------------
    # Nonparametric model: normalized ICAR
    # ----------------------------------------------------------------
    def nonparametric_model(event_bins, inj_bins, skip=False):

        if skip:
            log_R0 = numpyro.sample(
                "log_rate",
                dist.ImproperUniform(dist.constraints.real, (), ()),
            )
            return log_R0, jnp.zeros_like(event_bins[0], dtype=float), jnp.zeros_like(inj_bins[0], dtype=float)

        log_R0 = numpyro.sample(
            "log_rate",
            dist.ImproperUniform(dist.constraints.real, (), ()),
        )

        icar = ICAR_model(
            single_dimension_adj_matrices=pixelpop_data.adj_matrices,
            is_sparse=True,
            dx=jnp.exp(jnp.sum(pixelpop_data.logdV)), #here
        )

        interpolant = numpyro.sample("interpolant", icar)

        numpyro.deterministic(
            "interpolant_normalization",
            LSE(interpolant) + jnp.sum(pixelpop_data.logdV),
        )

        if pixelpop_data.lower_triangular:
            # Map LT vector to full mirrored bins-by-bins grid for indexing.
            interpolant_full = numpyro.deterministic(
                "interpolant_full", lt_map_fn(interpolant)
            )
            # Per-axis log marginals are skipped here because the mirrored
            # grid double-counts off-diagonal LT entries; compute post-hoc
            # if needed.
            interpolant_for_lookup = interpolant_full
        else:
            for ii, p in enumerate(pixelpop_data.pixelpop_parameters):
                sum_axes = tuple(
                    np.arange(pixelpop_data.dimension)[
                        np.r_[0:ii, ii + 1 : pixelpop_data.dimension]
                    ]
                )
                numpyro.deterministic(
                    f"log_marginal_{p}",
                    LSE(interpolant, axis=sum_axes)
                    + jnp.sum(pixelpop_data.logdV[:ii])
                    + jnp.sum(pixelpop_data.logdV[ii + 1 :]),
                )
            interpolant_for_lookup = interpolant

        # Recover sigma here. icar.quad_form handles lt_map internally for
        # the LT class, so we always pass the sampled `interpolant` directly.
        quad = icar.quad_form(interpolant)
        unscaled_gamma = numpyro.sample(
            "unscaled_gamma",
            numpyro.distributions.Gamma(
                concentration=((normalization_dof - 1) / 2)
            ),
        )
        precision = unscaled_gamma * quad / 2
        numpyro.deterministic("lnsigma", -0.5 * jnp.log(precision))

        # Return log p_PP at event/injection bin locations
        event_weights_PP = interpolant_for_lookup[event_bins]
        inj_weights_PP = interpolant_for_lookup[inj_bins]

        if log == "debug":
            jaxprint(
                "[DEBUG] pixelpop LSE(ew_PP)={ew}, LSE(iw_PP)={iw}",
                ew=LSE(event_weights_PP),
                iw=LSE(inj_weights_PP),
            )

        return log_R0, event_weights_PP, inj_weights_PP


    # ----------------------------------------------------------------
    # Full probabilistic model 
    # ----------------------------------------------------------------
    def probabilistic_model(posteriors, injections):
        base_event = posteriors["ln_dVTc"] - posteriors["log_prior"]
        base_inj = injections["ln_dVTc"] - injections["log_prior"]

        # Parametric:
        (
            common_param_event_weights,
            common_param_inj_weights,
            strong_param_event_weights,
            strong_param_inj_weights,
        ) = parametric_model(posteriors, injections)

        # Case 1: skip PixelPop
        if pixelpop_data.skip_nonparametric:
            log_R0 = numpyro.sample(
                "log_rate",
                dist.ImproperUniform(dist.constraints.real, (), ()),
            )
            event_weights = (
                base_event + log_R0
                + common_param_event_weights
                + strong_param_event_weights
            )
            inj_weights = (
                base_inj + log_R0
                + common_param_inj_weights
                + strong_param_inj_weights
            )

        # Case 2: PixelPop only (no mixture)
        elif pixelpop_data.skip_mixture:
            log_R0, event_weights_PP, inj_weights_PP = nonparametric_model(
                pixelpop_data.event_bins,
                pixelpop_data.inj_bins,
                skip=False,
            )
            event_weights = (
                base_event + log_R0
                + event_weights_PP
                + common_param_event_weights
            )
            inj_weights = (
                base_inj + log_R0
                + inj_weights_PP
                + common_param_inj_weights
            )

        # Case 3: Full mixture 
        else:
            log_R0, event_weights_PP, inj_weights_PP = nonparametric_model(
                pixelpop_data.event_bins,
                pixelpop_data.inj_bins,
                skip=False,
            )

            xi_args, xi_dist = pixelpop_data.xi_prior
            xi = numpyro.sample("xi", xi_dist(*xi_args))

            # Both sides are now log-probability-densities:
            event_weights = (
                base_event
                + common_param_event_weights
                + log_R0
                + jnp.logaddexp(
                    jnp.log(xi) + event_weights_PP,
                    jnp.log1p(-xi) + strong_param_event_weights,
                )
            )
            inj_weights = (
                base_inj
                + common_param_inj_weights
                + log_R0
                + jnp.logaddexp(
                    jnp.log(xi) + inj_weights_PP,
                    jnp.log1p(-xi) + strong_param_inj_weights,
                )
            )

        # --- Poisson rate likelihood ---
        ln_likelihood, nexp, pe_var, vt_var, total_var = rate_likelihood(
            event_weights,
            inj_weights,
            injections["total_generated"],
            live_time=injections["analysis_time"],
        )

        steep = 0.1 # Default is 0.1, value with which it works in gwtc3 is 0.01
        #jaxprint(f'Steepnes: {steep}')
        taper = smooth(total_var, pixelpop_data.UncertaintyCut ** 2, steep)

        numpyro.deterministic("log_likelihood", ln_likelihood)
        numpyro.deterministic("log_likelihood_variance", total_var)
        numpyro.deterministic("pe_variance", pe_var)
        numpyro.deterministic("vt_variance", vt_var)
        numpyro.deterministic("Nexp", nexp)

        numpyro.factor("log_likelihood_plus_taper", ln_likelihood + taper)

    return probabilistic_model, initial_value


def prior_probabilistic_model(pixelpop_data, log='default'):
    """
    Construct a hierarchical probabilistic model for GW population inference.

    This function sets up both parametric and nonparametric (CAR/ICAR) components
    of a gravitational-wave population model, returning a NumPyro-compatible model
    along with suitable initial values for MCMC warmup.

    Parameters
    ----------
    posteriors : dict
        Posterior samples keyed by parameter name. Each entry is shaped
        (Nobs, Nsample). Must also include 'ln_prior'.
    injections : dict
        Injection data keyed by parameter name. Each entry is shaped (Nfound).
        Must include 'ln_prior', 'total_generated' (int/float), and
        'analysis_time' (float).
    parameters : list of str
        Parameters for the nonparametric pixelized model (e.g., ["mass_1", "chi_eff"]).
    other_parameters : list of str
        Additional parameters modeled with parametric forms.
    bins : int or list of int
        Number of bins along each axis in the pixelized model.
    length_scales : bool, optional TODO!!!!
        If True, use independent CAR coupling parameters per axis.
    minima : dict, optional
        Mapping of parameter → minimum value. Defaults to typical BBH values.
    maxima : dict, optional
        Mapping of parameter → maximum value. Defaults to typical BBH values.
    parametric_models : dict, optional
        Mapping of parameter → callable defining parametric model.
    hyperparameters : dict, optional
        Mapping of parameter → list of hyperparameter names for its parametric model.
    priors : dict, optional
        Mapping of hyperparameter → (args, distribution) prior specification.
    random_initialization : bool, optional
        If True, initialize ICAR model with random noise instead of plausible values.
    lower_triangular : bool, optional
        If True, enforce p1 > p2 triangular support (used for joint m1–m2 models).
    constraint_funcs : list of callables, optional
        Extra constraint functions applied to hyperparameters.
    log : {"default", "debug"}, optional
        Logging verbosity.
        
    Returns
    -------
    probabilistic_model : callable
        NumPyro-compatible probabilistic model.
    initial_value : dict
        Suggested initial values for MCMC warmup.
    """

    posteriors = pixelpop_data.posteriors
    injections = pixelpop_data.injections
    parameters = pixelpop_data.pixelpop_parameters
    other_parameters = pixelpop_data.other_parameters
    bins = pixelpop_data.bins
    length_scales = pixelpop_data.length_scales
    minima = pixelpop_data.minima
    maxima = pixelpop_data.maxima
    parametric_models = pixelpop_data.parametric_models
    hyperparameters = pixelpop_data.parameter_to_hyperparameters
    priors = pixelpop_data.priors
    lower_triangular = pixelpop_data.lower_triangular
    constraint_funcs = pixelpop_data.constraint_funcs
    coupling_prior = pixelpop_data.coupling_prior

    dimension = len(parameters)
    if np.ndim(bins) == 0:
        bins = [bins] * dimension
    adj_matrices = [create_CAR_coupling_matrix(bins[ii], 1, isVisible=False) for ii in range(dimension)]
    
    if 'redshift' in parameters:
        from astropy.cosmology import Planck15
        from astropy import units
        max_z = np.maximum(np.max(injections['redshift']), np.max(posteriors['redshift']))
        zs = np.linspace(1e-6, max_z, 10000)
        dVs = Planck15.differential_comoving_volume(zs) * 4 * np.pi * units.sr
        ln_dVTc = np.log(dVs.to(units.Gpc**3).value) - np.log(1 + zs)
        event_z = posteriors['redshift']
        inj_z = injections['redshift']
        event_ln_dVTc = jnp.array(np.interp(event_z, zs, ln_dVTc))
        inj_ln_dVTc = jnp.array(np.interp(inj_z, zs, ln_dVTc))
    else:
        event_ln_dVTc = jnp.zeros_like(posteriors['log_prior'])
        inj_ln_dVTc = jnp.zeros_like(injections['log_prior'])
        
    event_bins, inj_bins, bin_axes, logdV, eprior, iprior = place_in_bins(parameters, posteriors, injections, bins=bins, minima=minima, maxima=maxima)
    
    posteriors['log_prior'] += eprior
    injections['log_prior'] += iprior

    # update models
    parameter_to_hyperparameters = gwparameter_to_hyperparameters.copy()
    parameter_to_hyperparameters.update(hyperparameters)

    parameter_to_gwpop_model = {}
    for p in other_parameters:
        if p in parametric_models:
            print(f'Updating {p} model from {gwparameter_to_model[p].__name__} to {parametric_models[p].__name__}')
            print(f'\t ...with hyperparameters {parameter_to_hyperparameters[p]}')
            parameter_to_gwpop_model[p] = parametric_models[p]
        else:
            print(f'Using default {p} model {gwparameter_to_model[p].__name__}')
            parameter_to_gwpop_model[p] = gwparameter_to_model[p]

    # default priors
    hyperparameter_priors = {}
    for p in other_parameters:
        for h in parameter_to_hyperparameters[p]:
            if h in priors:    
                pprint = priors[h]
                print(f'Using custom prior {h} = {pprint[1].__name__}({str(pprint[0])[1:-1]}) in {p} model')
                hyperparameter_priors[h] = priors[h]
            else:
                pprint = default_priors[h]
                print(f'Using default prior {h} = {pprint[1].__name__}({str(pprint[0])[1:-1]}) in {p} model')
                hyperparameter_priors[h] = default_priors[h]

    if lower_triangular:
        lt_map = lower_triangular_map(bins[0])
        tri_size = int(bins[0]*(bins[0]+1)/2) 
        unique_sample_shape = (tri_size,) + tuple(bins[2:])
        # lower triangular in first two dimensions
    else:
        unique_sample_shape = bins
    normalization_dof = np.prod(unique_sample_shape)

    def get_initial_value():
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
        return_dict = {'_eigenbasis_sites': jnp.array(
            np.random.normal(loc=0, scale=1, size=unique_sample_shape))
            }
        return return_dict
            
    initial_value = get_initial_value()

    def parametric_prior(data, injections, event_weights, inj_weights):
        """
        Evaluate the parametric population model contribution.

        Draws hyperparameters from their priors
        """
        sample = {}
        for key in hyperparameter_priors:
            args, distribution = hyperparameter_priors[key]
            if distribution.__name__ == 'Delta':
                sample[key] = args[0]
            else:
                sample[key] = numpyro.sample(key, distribution(*args))        
        for constraint_func in constraint_funcs:
            numpyro.factor(constraint_func.__name__, constraint_func(sample))
            if log == 'debug':
                jaxprint('[DEBUG] constraint functions:', constraint_func.__name__, constraint_func(sample))

        for p in other_parameters:
            event_weights += parameter_to_gwpop_model[p](data, *[sample[h] for h in parameter_to_hyperparameters[p]])
            inj_weights += parameter_to_gwpop_model[p](injections, *[sample[h] for h in parameter_to_hyperparameters[p]])
            if log == 'debug':
                jaxprint('[DEBUG] parametric {p} LSE(event_weights)={ew}, LSE(injection_weights)={iw}', p=p, ew=LSE(event_weights), iw=LSE(inj_weights))
                if not jnp.isfinite(LSE(event_weights)):
                    for parameter in map_to_gwpop_parameters[p]:
                        jaxprint('[DEBUG] inf event weights at {p}={d}', p=parameter, d=data[parameter][jnp.where(event_weights == jnp.inf)])
                if not jnp.isfinite(LSE(inj_weights)):
                    for parameter in map_to_gwpop_parameters[p]:
                        jaxprint('[DEBUG] inf injection weights at {p}={d}', p=parameter, d=injections[parameter][jnp.where(inj_weights == jnp.inf)])
        return event_weights, inj_weights
    def nonparametric_prior(event_bins, inj_bins, event_weights, inj_weights):
        """
        Evaluate the nonparametric (ICAR/CAR) pixelized model contribution.
        """
        if length_scales:
            # TODO!!
            lsigma = numpyro.sample('lnsigma', coupling_prior[1](*coupling_prior[0]), sample_shape=(dimension,))
        else:
            lsigma = numpyro.sample('lnsigma', coupling_prior[1](*coupling_prior[0]), sample_shape=()) 

        mask = jnp.ones(bins, dtype=bool).at[(0,) * len(unique_sample_shape)].set(False)

        _eigenbasis_sites = numpyro.sample(
            "_eigenbasis_sites",
            dist.Normal(0., 1.).expand(unique_sample_shape).mask(mask)
        )
        # _eigenbasis_site_0 = numpyro.sample("_eigenbasis_site_0", dist.ImproperUniform(dist.constraints.real, (), ()))
        _eigenbasis_site_0 = 0.
        eigenbasis_sites = _eigenbasis_sites.at[(0,) * dimension].set(_eigenbasis_site_0)
        
        if lower_triangular:
            eigenbasis_sites = lower_triangular_map(eigenbasis_sites)

        merger_rate_density = numpyro.deterministic(
            'merger_rate_density',
            DiagonalizedICARTransform(lsigma, adj_matrices, is_sparse=True)(
                eigenbasis_sites
            )
        )
        if not lower_triangular:
            normalization = numpyro.deterministic('log_rate', LSE(merger_rate_density)+jnp.sum(logdV))
            for ii, p in enumerate(parameters):
                sum_axes = tuple(np.arange(dimension)[np.r_[0:ii,ii+1:dimension]])
                numpyro.deterministic(f'log_marginal_{p}', LSE(merger_rate_density-normalization, axis=sum_axes) + jnp.sum(logdV[:ii]) + jnp.sum(logdV[ii+1:]))

        event_weights += merger_rate_density[event_bins] # (69,3194)
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
        event_weights, inj_weights = nonparametric_prior(event_bins, inj_bins, event_ln_dVTc-posteriors['log_prior'], inj_ln_dVTc-injections['log_prior'])
        event_weights, inj_weights = parametric_prior(posteriors, injections, event_weights, inj_weights)

        ln_likelihood, nexp, pe_var, vt_var, total_var = rate_likelihood(event_weights, inj_weights, injections['total_generated'], live_time=injections['analysis_time'])

        # save these values!
        numpyro.deterministic("log_likelihood", ln_likelihood)
        numpyro.deterministic("log_likelihood_variance", total_var)
        numpyro.deterministic("pe_variance", pe_var)
        numpyro.deterministic("vt_variance", vt_var)
        numpyro.deterministic("Nexp", nexp)

    return probabilistic_model, initial_value