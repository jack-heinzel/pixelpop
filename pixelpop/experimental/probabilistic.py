import numpy as np
from ..utils.nearest_neighbor import create_CAR_coupling_matrix
from ..models.gwpop_models import * 
from ..models.car import ICAR_length_scales, lower_triangular_map, lower_triangular_log_prob, mult_outer, add_outer
from .car import DiagonalizedICARTransform, sigma_marginalized_ICAR, lower_triangular_sigma_marg_log_prob, lower_triangular_sigma_marg_log_prob_and_log_quad,
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.debug import print as jaxprint
from jax.nn import log_sigmoid
from ..utils.data import place_in_bins
from jax.scipy.special import logsumexp as LSE
from ..utils.data import place_in_bins
import numpyro

def _bin_samples_on_mixture_grid(mixture_config, posteriors, injections):
    """
    Bin posterior and injection samples onto the mixture's ICAR grid.

    Separate from the main PixelPopData binning because the mixing fraction
    grid may use different parameters or bin counts.

    Parameters
    ----------
    mixture_config : MixtureConfig
    posteriors, injections : dict

    Returns
    -------
    event_bins : tuple of ndarray
        Bin indices for events on the mixture grid.
    inj_bins : tuple of ndarray
        Bin indices for injections on the mixture grid.
    bin_axes : list of ndarray
        Bin edges for each grid parameter.
    """
    event_bins, inj_bins, bin_axes, _, _, _ = place_in_bins(
        mixture_config.grid_parameters,
        posteriors,
        injections,
        bins=mixture_config.grid_bins,
        minima=mixture_config.grid_minima,
        maxima=mixture_config.grid_maxima,
    )
    return event_bins, inj_bins, bin_axes

def setup_probabilistic_model_with_mixture(
    pixelpop_data, mixture_config, log='default'
):
    """
    Construct a hierarchical model with parametric components plus a
    pixelpopped mixing fraction between two parametric sub-models.

    The full model factorizes as::

        prod_i p_i(theta_i)
            * [xi(theta_grid) * p_A(theta_mix)
               + (1 - xi(theta_grid)) * p_B(theta_mix)]

    Parameters
    ----------
    pixelpop_data : PixelPopData
        Standard pixelpop data container. The mixture target parameter(s)
        should NOT appear in other_parameters.
    mixture_config : MixtureConfig
        Configuration for the pixelpopped mixing fraction.
    log : str, optional
        Logging level ('default' or 'debug').

    Returns
    -------
    probabilistic_model : callable
        NumPyro-compatible model function.
    initial_value : dict
        Suggested initial values for MCMC warmup.
    """

    # =================================================================
    # Pre-compute mixture grid binning
    # =================================================================
    mix_event_bins, mix_inj_bins, mix_bin_axes = \
        _bin_samples_on_mixture_grid(
            mixture_config,
            pixelpop_data.posteriors,
            pixelpop_data.injections,
        )

    mix_name = mixture_config.name
    normalization_dof = mixture_config.normalization_dof
    mix_dim = mixture_config.dimension
    grid_shape = tuple(mixture_config.grid_bins)

    # =================================================================
    # ICAR model choice
    # =================================================================
    if mixture_config.marginalize_sigma:
        ICAR_model = sigma_marginalized_ICAR
    else:
        ICAR_model = ICAR_length_scales

    # =================================================================
    # Initial values: phi=0  =>  xi = sigmoid(0) = 0.5 everywhere
    # =================================================================
    initial_value = {f'{mix_name}_field': jnp.zeros(grid_shape)}

    # =================================================================
    # Parametric model (unchanged — handles m1, q, z, etc.)
    # =================================================================
    def parametric_model(data, injections, event_weights, inj_weights):
        """
        Evaluate all standard parametric contributions from
        pixelpop_data.other_parameters.

        Returns updated weights AND the sampled hyperparameters dict
        (so that shared hyperparameters can be reused downstream).
        """
        sample = {}
        for key in pixelpop_data.priors:
            args, distribution = pixelpop_data.priors[key]
            if distribution.__name__ == 'Delta':
                sample[key] = args[0]
            else:
                sample[key] = numpyro.sample(key, distribution(*args))

        if log == 'debug':
            for p in pixelpop_data.other_parameters:
                jaxprint('[DEBUG] =================================')
                jaxprint('[DEBUG] parametric parameters: {p}', p=p)
                for k in pixelpop_data.parameter_to_hyperparameters[p]:
                    jaxprint('[DEBUG] \t {k} sample: {s}', k=k, s=sample[k])

        for constraint_func in pixelpop_data.constraint_funcs:
            numpyro.factor(constraint_func.__name__, constraint_func(sample))

        for p in pixelpop_data.other_parameters:
            event_weights += pixelpop_data.parametric_models[p](
                data,
                *[sample[h] for h in pixelpop_data.parameter_to_hyperparameters[p]],
            )
            inj_weights += pixelpop_data.parametric_models[p](
                injections,
                *[sample[h] for h in pixelpop_data.parameter_to_hyperparameters[p]],
            )
            if log == 'debug':
                jaxprint(
                    '[DEBUG] parametric {p} LSE(ew)={ew}, LSE(iw)={iw}',
                    p=p, ew=LSE(event_weights), iw=LSE(inj_weights),
                )

        return event_weights, inj_weights, sample

    # =================================================================
    # Sample mixture-specific hyperparameters
    # =================================================================
    def sample_mixture_hyperparameters():
        """
        Draw hyperparameters for both mixture components from their
        priors. Returns dict of sampled values.
        """
        mix_sample = {}
        for key, (args, distribution) in mixture_config.all_priors.items():
            if distribution.__name__ == 'Delta':
                mix_sample[key] = args[0]
            else:
                mix_sample[key] = numpyro.sample(key, distribution(*args))
        return mix_sample

    # =================================================================
    # Mixing fraction ICAR model
    # =================================================================
    def mixing_fraction_icar(skip=False):
        """
        Sample the ICAR field and convert to log-mixing-fractions
        via sigmoid.

        Parameters
        ----------
        skip : bool
            If True, use a single global mixing fraction instead
            of the spatially-varying field.

        Returns
        -------
        log_xi_field : jnp.ndarray or scalar
            log(sigmoid(phi)) on the grid (or scalar if skip).
        log_1mxi_field : jnp.ndarray or scalar
            log(1 - sigmoid(phi)).
        """
        if skip:
            logit_xi = numpyro.sample(
                f'{mix_name}_logit_xi_global',
                dist.Normal(0.0, 2.0),
            )
            log_xi = log_sigmoid(logit_xi)
            log_1mxi = log_sigmoid(-logit_xi)
            numpyro.deterministic(
                f'{mix_name}_xi_global', jnp.sigmoid(logit_xi),
            )
            return log_xi, log_1mxi

        # --- ICAR coupling ---
        if not mixture_config.marginalize_sigma:
            cp = mixture_config.coupling_prior
            if mixture_config.length_scales:
                lsigma = numpyro.sample(
                    f'{mix_name}_lnsigma',
                    cp[1](*cp[0]),
                    sample_shape=(mix_dim,),
                )
            else:
                lsigma = numpyro.sample(
                    f'{mix_name}_lnsigma',
                    cp[1](*cp[0]),
                    sample_shape=(),
                )

        # --- Sample the field ---
        if mixture_config.marginalize_sigma:
            icar = ICAR_model(
                single_dimension_adj_matrices=mixture_config.adj_matrices,
                is_sparse=True,
            )
            mix_field = numpyro.sample(
                f'{mix_name}_field',
                dist.ImproperUniform(
                    dist.constraints.real, grid_shape, (),
                ),
            )
            prior_factor, quad = icar.log_prob_and_quad(mix_field)
            numpyro.factor(f'{mix_name}_prior_factor', prior_factor)
        else:
            mix_field = numpyro.sample(
                f'{mix_name}_field',
                ICAR_model(
                    log_sigmas=lsigma,
                    single_dimension_adj_matrices=mixture_config.adj_matrices,
                    is_sparse=True,
                ),
            )

        # --- Sigma marginalization ---
        if mixture_config.marginalize_sigma:
            unscaled_gamma = numpyro.sample(
                f'{mix_name}_unscaled_gamma',
                numpyro.distributions.Gamma(
                    concentration=(normalization_dof / 2),
                ),
            )
            precision = 2 * unscaled_gamma / quad
            numpyro.deterministic(
                f'{mix_name}_lnsigma', -0.5 * jnp.log(precision),
            )

        # --- Convert to mixing fractions ---
        xi_field = numpyro.deterministic(
            f'{mix_name}_xi_field', jnp.sigmoid(mix_field),
        )
        log_xi_field = log_sigmoid(mix_field)
        log_1mxi_field = log_sigmoid(-mix_field)

        # --- Marginal diagnostics ---
        for ii, p in enumerate(mixture_config.grid_parameters):
            sum_axes = tuple(
                np.arange(mix_dim)[np.r_[0:ii, ii + 1 : mix_dim]]
            )
            numpyro.deterministic(
                f'{mix_name}_marginal_xi_{p}',
                jnp.mean(xi_field, axis=sum_axes),
            )

        return log_xi_field, log_1mxi_field

    # =================================================================
    # Apply mixture to event/injection weights
    # =================================================================
    def apply_mixture(
        event_weights, inj_weights, mix_sample,
        log_xi_field, log_1mxi_field, data, injections,
        skip=False,
    ):
        """
        Compute the mixture log-probability for events and injections
        and add it to the running weights.

        For each sample:
            log[ xi(bin) * p_A(data) + (1 - xi(bin)) * p_B(data) ]

        Parameters
        ----------
        event_weights, inj_weights : jnp.ndarray
            Running log-weights.
        mix_sample : dict
            Sampled hyperparameters for mixture components.
        log_xi_field, log_1mxi_field : jnp.ndarray or scalar
            Log mixing fractions (grid or scalar).
        data, injections : dict
            Event posterior samples / injection data.
        skip : bool
            Whether the ICAR was skipped (scalar xi).

        Returns
        -------
        event_weights, inj_weights : jnp.ndarray
        """
        # Evaluate both component models on events and injections
        log_pA_ev = mixture_config.component_a_model(
            data,
            *[mix_sample[h] for h in mixture_config.component_a_hyperparameters],
        )
        log_pB_ev = mixture_config.component_b_model(
            data,
            *[mix_sample[h] for h in mixture_config.component_b_hyperparameters],
        )
        log_pA_ij = mixture_config.component_a_model(
            injections,
            *[mix_sample[h] for h in mixture_config.component_a_hyperparameters],
        )
        log_pB_ij = mixture_config.component_b_model(
            injections,
            *[mix_sample[h] for h in mixture_config.component_b_hyperparameters],
        )

        if skip:
            log_xi_ev = log_xi_field
            log_1mxi_ev = log_1mxi_field
            log_xi_ij = log_xi_field
            log_1mxi_ij = log_1mxi_field
        else:
            # Look up per-sample mixing fractions from the grid
            log_xi_ev = log_xi_field[mix_event_bins]
            log_1mxi_ev = log_1mxi_field[mix_event_bins]
            log_xi_ij = log_xi_field[mix_inj_bins]
            log_1mxi_ij = log_1mxi_field[mix_inj_bins]

        # Mixture in log-space
        event_weights += jnp.logaddexp(
            log_xi_ev + log_pA_ev,
            log_1mxi_ev + log_pB_ev,
        )
        inj_weights += jnp.logaddexp(
            log_xi_ij + log_pA_ij,
            log_1mxi_ij + log_pB_ij,
        )

        if log == 'debug':
            jaxprint(
                '[DEBUG] {n} mixture LSE(ew)={ew}, LSE(iw)={iw}',
                n=mix_name, ew=LSE(event_weights), iw=LSE(inj_weights),
            )

        return event_weights, inj_weights

    # =================================================================
    # Full probabilistic model
    # =================================================================
    def probabilistic_model(posteriors, injections):
        """
        Full probabilistic model combining:
          1. Parametric contributions (m1, q, z, etc.)
          2. Pixelpopped mixing fraction between two sub-models
          3. Rate likelihood

        Parameters
        ----------
        posteriors : dict
            Posterior samples from detected events.
        injections : dict
            Injection data including selection effects.
        """
        event_weights = posteriors['ln_dVTc'] - posteriors['log_prior']
        inj_weights = injections['ln_dVTc'] - injections['log_prior']

        # 1) Standard parametric models
        event_weights, inj_weights, sample = parametric_model(
            posteriors, injections, event_weights, inj_weights,
        )

        # 2) Sample mixture hyperparameters
        mix_sample = sample_mixture_hyperparameters()

        # 3) ICAR mixing fraction field
        skip = pixelpop_data.skip_nonparametric
        log_xi_field, log_1mxi_field = mixing_fraction_icar(skip=skip)

        # 4) Apply mixture to weights
        event_weights, inj_weights = apply_mixture(
            event_weights, inj_weights, mix_sample,
            log_xi_field, log_1mxi_field,
            posteriors, injections, skip=skip,
        )

        # 5) Rate likelihood
        ln_likelihood, nexp, pe_var, vt_var, total_var = \
            rate_likelihood(
                event_weights,
                inj_weights,
                injections['total_generated'],
                live_time=injections['analysis_time'],
            )
        taper = smooth(
            total_var, pixelpop_data.UncertaintyCut ** 2, 0.1,
        )

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