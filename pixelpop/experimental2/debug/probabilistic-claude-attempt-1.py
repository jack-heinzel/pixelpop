import numpy as np
from ..utils.nearest_neighbor import create_CAR_coupling_matrix
from ..models.gwpop_models import * 
from ..models.car import lower_triangular_map, mult_outer, add_outer
from .car import (DiagonalizedICARTransform, initialize_ICAR_normalized)
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.debug import print as jaxprint
from ..utils.data import place_in_bins
from jax.scipy.special import logsumexp as LSE
import numpyro


def setup_mixture_probabilistic_model(pixelpop_data, log="default"):
    """
    Construct a mixture probabilistic model for GW population inference.

    The model is:

        R(θ) = R₀ · [ ξ · p_PP(θ_pixel) + (1-ξ) · ∏ pᵢ(θ_pixel_i) ]
                   · ∏ⱼ pⱼ(θ_other_j)

    where p_PP is a normalized ICAR (σ-marginalized).

    The normalized ICAR is implemented via:
      1. Sample unconstrained ``phi`` from ``ImproperUniform``
      2. Normalize: ``interpolant = phi - LSE(phi) - log(dV)``
      3. Evaluate ``ICAR_normalized.log_prob(interpolant)`` as a factor
         (the same class used in the toy notebook)

    Parameters
    ----------
    pixelpop_data : MixturePixelPopData
        Configuration dataclass.
    log : str, optional
        Logging level. ``'debug'`` enables verbose JAX printing.

    Returns
    -------
    probabilistic_model : callable
        NumPyro model function ``(posteriors, injections) -> None``.
    initial_value : dict
        Initial values for MCMC warmup.
    """

    pp_dim = len(pixelpop_data.pixelpop_parameters)
    n_pixels = int(np.prod(pixelpop_data.bins))
    log_dV = float(jnp.sum(pixelpop_data.logdV))
    dV = float(jnp.exp(log_dV))

    # ----------------------------------------------------------------
    # Build the ICAR_normalized instance (same class as the toy notebook)
    # ----------------------------------------------------------------
    ICAR_norm_cls = initialize_ICAR_normalized(pp_dim)
    icar_dist = ICAR_norm_cls(
        single_dimension_adj_matrices=pixelpop_data.adj_matrices,
        is_sparse=True,
        dx=dV,
    )

    # ----------------------------------------------------------------
    # Initial values
    # ----------------------------------------------------------------
    # phi is unconstrained; we initialize with small noise around zero.
    # After normalization: interpolant = phi - LSE(phi) - log_dV
    # When phi ≈ const, interpolant ≈ -log(n_pixels) - log_dV (uniform).
    init_phi = jnp.array(
        np.random.normal(loc=0, scale=0.1, size=tuple(pixelpop_data.bins))
    )

    initial_value = {
        "phi": init_phi,
        "log_rate": jnp.log(50.0),
    }

    # ----------------------------------------------------------------
    # Parametric model for "common strong" parameters
    # ----------------------------------------------------------------
    def common_parametric_model(data, injections, event_weights, inj_weights):
        """Evaluate common parametric models that multiply the entire mixture."""
        sample = {}
        for p in pixelpop_data.common_strong_parameters:
            for h in pixelpop_data.parameter_to_hyperparameters[p]:
                if h not in sample:
                    args, distribution = pixelpop_data.priors[h]
                    if distribution.__name__ == "Delta":
                        sample[h] = args[0]
                    else:
                        sample[h] = numpyro.sample(h, distribution(*args))

        for constraint_func in pixelpop_data.constraint_funcs:
            numpyro.factor(constraint_func.__name__, constraint_func(sample))

        for p in pixelpop_data.common_strong_parameters:
            model_fn = pixelpop_data.parametric_models[p]
            hypers = [
                sample[h] for h in pixelpop_data.parameter_to_hyperparameters[p]
            ]
            event_weights += model_fn(data, *hypers)
            inj_weights += model_fn(injections, *hypers)

            if log == "debug":
                jaxprint(
                    "[DEBUG] common param {p}: LSE(ew)={ew}, LSE(iw)={iw}",
                    p=p, ew=LSE(event_weights), iw=LSE(inj_weights),
                )

        return event_weights, inj_weights, sample

    # ----------------------------------------------------------------
    # Mixture parametric model for pixel dimensions
    # ----------------------------------------------------------------
    def mixture_parametric_model(data, injections, already_sampled):
        """
        Evaluate independent parametric marginals for the pixel dimensions.
        Returns log p_param(θ_pixel) = Σᵢ log pᵢ(θ_pixel_i).
        """
        sample = dict(already_sampled)

        for p in pixelpop_data.mixture_parameters:
            for h in pixelpop_data.mixture_parameter_to_hyperparameters[p]:
                if h not in sample:
                    args, distribution = pixelpop_data.mixture_priors[h]
                    if distribution.__name__ == "Delta":
                        sample[h] = args[0]
                    else:
                        sample[h] = numpyro.sample(h, distribution(*args))

        log_p_events = jnp.zeros_like(data["log_prior"])
        log_p_inj = jnp.zeros_like(injections["log_prior"])

        for p in pixelpop_data.mixture_parameters:
            model_fn = pixelpop_data.mixture_parametric_models[p]
            hypers = [
                sample[h]
                for h in pixelpop_data.mixture_parameter_to_hyperparameters[p]
            ]
            log_p_events += model_fn(data, *hypers)
            log_p_inj += model_fn(injections, *hypers)

            if log == "debug":
                jaxprint(
                    "[DEBUG] mixture param {p}: LSE(ew)={ew}, LSE(iw)={iw}",
                    p=p, ew=LSE(log_p_events), iw=LSE(log_p_inj),
                )

        return log_p_events, log_p_inj

    # ----------------------------------------------------------------
    # Full probabilistic model
    # ----------------------------------------------------------------
    def probabilistic_model(posteriors, injections):
        """
        Full mixture probabilistic model for hierarchical GW population
        inference.

        The log-likelihood per event (per PE sample) is:

            log R₀ + log[ξ · exp(interpolant[bin]) + (1-ξ) · ∏ pᵢ(θᵢ)]
                   + Σⱼ log pⱼ(θⱼ_other)
                   + log(dVc/dz / (1+z))
                   - log π(θ)
        """

        # --- Common parametric part (multiplies entire mixture) ---
        event_weights = posteriors["ln_dVTc"] - posteriors["log_prior"]
        inj_weights = injections["ln_dVTc"] - injections["log_prior"]

        event_weights, inj_weights, shared_sample = common_parametric_model(
            posteriors, injections, event_weights, inj_weights
        )

        # --- Overall rate ---
        log_R0 = numpyro.sample(
            "log_rate",
            dist.ImproperUniform(dist.constraints.real, (), ()),
        )

        # --- Mixing fraction ---
        xi_args, xi_dist = pixelpop_data.xi_prior
        xi = numpyro.sample("xi", xi_dist(*xi_args))

        # =============================================================
        # Normalized PixelPop via ImproperUniform + ICAR_normalized.log_prob
        #
        # This follows the same pattern as the existing PixelPop codebase
        # (ImproperUniform + factor) but uses your ICAR_normalized class
        # from the toy notebook for the prior evaluation.
        # =============================================================

        # Step 1: Sample unconstrained field phi
        phi = numpyro.sample(
            "phi",
            dist.ImproperUniform(
                dist.constraints.real,
                tuple(pixelpop_data.bins),
                (),
            ),
        )

        # Step 2: Normalize to get interpolant (a proper log-density)
        #   interpolant = phi - LSE(phi) - log_dV
        #   => LSE(interpolant) = -log_dV  (exact by construction)
        #   => sum(exp(interpolant)) * dV = 1
        lse_phi = LSE(phi)
        interpolant = numpyro.deterministic(
            "interpolant", phi - lse_phi - log_dV
        )

        # Step 3: ICAR σ-marginalized prior on the normalized field
        #   Uses the same ICAR_normalized.log_prob as the toy notebook.
        icar_log_prob = icar_dist.log_prob(interpolant)
        numpyro.factor("icar_prior", icar_log_prob)

        # Diagnostics: verify normalization (should be ≈ 0 by construction)
        normalization = LSE(interpolant) + jnp.sum(pixelpop_data.logdV)
        numpyro.deterministic("interpolant_normalization", normalization)

        if log == "debug":
            jaxprint(
                "[DEBUG] interpolant norm={n}, LSE(phi)={lse}, "
                "icar_log_prob={ilp}",
                n=normalization, lse=lse_phi, ilp=icar_log_prob,
            )

        # =============================================================
        # PixelPop log-density at event/injection bin locations
        # =============================================================
        log_pp_events = interpolant[pixelpop_data.event_bins]
        log_pp_inj = interpolant[pixelpop_data.inj_bins]

        # --- Parametric marginals for pixel dimensions ---
        log_param_events, log_param_inj = mixture_parametric_model(
            posteriors, injections, shared_sample
        )

        # --- Mixture in log-space ---
        log_xi = jnp.log(xi)
        log_1mxi = jnp.log1p(-xi)

        log_mix_events = jnp.logaddexp(
            log_xi + log_pp_events,
            log_1mxi + log_param_events,
        )
        log_mix_inj = jnp.logaddexp(
            log_xi + log_pp_inj,
            log_1mxi + log_param_inj,
        )

        # Add mixture + rate to the weights
        event_weights += log_R0 + log_mix_events
        inj_weights += log_R0 + log_mix_inj

        if log == "debug":
            jaxprint(
                "[DEBUG] post-mixture: LSE(ew)={ew}, LSE(iw)={iw}, xi={xi}",
                ew=LSE(event_weights), iw=LSE(inj_weights), xi=xi,
            )

        # --- Poisson rate likelihood ---
        ln_likelihood, nexp, pe_var, vt_var, total_var = rate_likelihood(
            event_weights,
            inj_weights,
            injections["total_generated"],
            live_time=injections["analysis_time"],
        )

        taper = smooth(total_var, pixelpop_data.UncertaintyCut ** 2, 0.1)

        # Store diagnostics
        numpyro.deterministic("log_likelihood", ln_likelihood)
        numpyro.deterministic("log_likelihood_variance", total_var)
        numpyro.deterministic("pe_variance", pe_var)
        numpyro.deterministic("vt_variance", vt_var)
        numpyro.deterministic("Nexp", nexp)

        # PixelPop marginals for diagnostics
        for ii, p in enumerate(pixelpop_data.pixelpop_parameters):
            sum_axes = tuple(
                np.arange(pp_dim)[np.r_[0:ii, ii + 1 : pp_dim]]
            )
            numpyro.deterministic(
                f"log_marginal_{p}",
                LSE(interpolant, axis=sum_axes)
                + jnp.sum(pixelpop_data.logdV[:ii])
                + jnp.sum(pixelpop_data.logdV[ii + 1 :]),
            )

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