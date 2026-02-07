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

def setup_mixture_probabilistic_model(pixelpop_data, log='default'):
    """
    Construct a hierarchical probabilistic model for GW population inference.

    This function sets up a mixture of parametric and nonparametric (CAR/ICAR) components
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
        # print(bin_med)
        interpolation_grid = np.meshgrid(*bin_med, indexing='ij')
        
        return_key = 'merger_rate_density'
        if random_initialization:
            if pixelpop_data.lower_triangular:
                return_dict = {'base_interpolation': jnp.array(
                    np.random.normal(loc=0, scale=1, size=unique_sample_shape)
                    )}
            else:
                init = jnp.array(np.random.normal(loc=0, scale=1, size=interpolation_grid[0].shape))
                dx = jnp.exp(jnp.sum(pixelpop_data.logdV))
                init = -jnp.log(jnp.prod(np.array(pixelpop_data.bin_axes))*dx) + init - LSE(init)
                return_dict = {return_key: init, 'log_rate_total': np.log(50)}
                # return_dict = {return_key: jnp.array(
                #     np.random.normal(loc=0, scale=1, size=interpolation_grid[0].shape))
                #     }
        
                
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
            pixelpop_data.posteriors['ln_dVTc'].shape[0], 
            pixelpop_data.injections['ln_dVTc']-pixelpop_data.injections['log_prior'],
            random_initialization=pixelpop_data.random_initialization
            )

    def parametric_model(data, injections):
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

        Returns
        -------
        event_weights : ndarray
            Updated event log-weights including parametric contributions.
        inj_weights : ndarray
            Updated injection log-weights including parametric contributions.
        """
        sample = {}

        common_param_event_weights = 0
        common_param_inj_weights = 0
        strong_param_event_weights = 0
        strong_param_inj_weights = 0
        
        for key in pixelpop_data.priors:
            args, distribution = pixelpop_data.priors[key]
            if distribution.__name__ == 'Delta':
                sample[key] = args[0]
            else:
                sample[key] = numpyro.sample(key, distribution(*args))        
        
        if log == 'debug':
            # SAL: in this case, we need to debug both other (common) and PixelPop 
            for p in pixelpop_data.other_parameters+pixelpop_data.pixelpop_parameters:
                jaxprint('[DEBUG] =================================')
                jaxprint('[DEBUG] parametric parameters: {p}', p=p)
                jaxprint('[DEBUG] =================================')       
                for k in pixelpop_data.parameter_to_hyperparameters[p]:
                    jaxprint('[DEBUG] \t {k} sample: {s}', k=k, s=sample[k])
        for constraint_func in pixelpop_data.constraint_funcs:
            numpyro.factor(constraint_func.__name__, constraint_func(sample))
            if log == 'debug':
                jaxprint('[DEBUG] constraint functions:', constraint_func.__name__, constraint_func(sample))
        
        # SAL: Handle common (i.e., other) parameters separately from parametric models used for PixelPop parameters.
        for p in pixelpop_data.other_parameters:
            common_param_event_weights += pixelpop_data.parametric_models[p](
                data, *[sample[h] for h in pixelpop_data.parameter_to_hyperparameters[p]]
                )
            common_param_inj_weights += pixelpop_data.parametric_models[p](
                injections, *[sample[h] for h in pixelpop_data.parameter_to_hyperparameters[p]]
                )
            if log == 'debug':
                jaxprint('[DEBUG] parametric {p} LSE(common_param_event_weights)={ew}, LSE(common_param_inj_weights)={iw}', p=p, ew=LSE(common_param_event_weights), iw=LSE(common_param_inj_weights))
                if not jnp.isfinite(LSE(common_param_event_weights)):
                    for parameter in pixelpop_data.parameter_to_hyperparameters[p]:
                        jaxprint('[DEBUG] inf common event weights at {pp}={d}', pp=parameter, d=data[parameter][jnp.where(common_param_event_weights == jnp.inf)])
                if not jnp.isfinite(LSE(common_param_inj_weights)):
                    for parameter in pixelpop_data.parameter_to_hyperparameters[p]:
                        jaxprint('[DEBUG] inf common injection weights at {pp}={d}', pp=parameter, d=injections[parameter][jnp.where(common_param_inj_weights == jnp.inf)])

        # SAL: Handle strong parametric models used for PixelPop parameters.
        for sp in pixelpop_data.pixelpop_parameters:
            strong_param_event_weights += pixelpop_data.parametric_models[sp](
                data, *[sample[h] for h in pixelpop_data.parameter_to_hyperparameters[sp]]
                )
            strong_param_inj_weights += pixelpop_data.parametric_models[sp](
                injections, *[sample[h] for h in pixelpop_data.parameter_to_hyperparameters[sp]]
                )
            if log == 'debug':
                jaxprint('[DEBUG] parametric {p} LSE(strong_param_event_weights)={ew}, LSE(strong_param_inj_weights)={iw}', p=sp, ew=LSE(strong_param_event_weights), iw=LSE(strong_param_inj_weights))
                if not jnp.isfinite(LSE(strong_param_event_weights)):
                    for parameter in pixelpop_data.parameter_to_hyperparameters[sp]:
                        jaxprint('[DEBUG] inf strong event weights at {pp}={d}', pp=parameter, d=data[parameter][jnp.where(strong_param_event_weights == jnp.inf)])
                if not jnp.isfinite(LSE(strong_param_inj_weights)):
                    for parameter in pixelpop_data.parameter_to_hyperparameters[sp]:
                        jaxprint('[DEBUG] inf strong injection weights at {pp}={d}', pp=parameter, d=injections[parameter][jnp.where(strong_param_inj_weights == jnp.inf)])
                       
        return common_param_event_weights, common_param_inj_weights, strong_param_event_weights, strong_param_inj_weights

    # SAL: only include mixture model
    ICAR_model = initialize_ICAR_normalized(pixelpop_data.dimension)    

    def nonparametric_model(event_bins, inj_bins, skip=False):
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
        skip : bool, optional
            If True, skip the ICAR model and use only a single log-rate parameter.

        Returns
        -------
        event_weights_PP: ndarray
            event log-weights including nonparametric contributions.
        inj_weights_PP: ndarray
            injection log-weights including nonparametric contributions.
        """

        if skip:
            R = numpyro.sample('log_rate', dist.ImproperUniform(dist.constraints.real, (), ()))
            # SAL: Remove comoving factor here since I will do it in probabilistic_model.
            # In this case there would be no mixture.
            return R[None,None], R[None]

        if pixelpop_data.lower_triangular:
            
            base_interpolation = numpyro.sample('base_interpolation', dist.ImproperUniform(dist.constraints.real, unique_sample_shape, ()))
            merger_rate_density = numpyro.deterministic('merger_rate_density', lt_map(base_interpolation))
            jaxprint('SAL TODO: implement these functions for lower triangular.')
            
            # if pixelpop_data.marginalize_sigma:
            #     prior_factor, quad = lower_triangular_sigma_marg_log_prob_and_log_quad(merger_rate_density, normalization_dof, pixelpop_data.adj_matrices)
            # else:
            #     prior_factor = lower_triangular_log_prob(merger_rate_density, normalization_dof, lsigma, pixelpop_data.adj_matrices)
            # numpyro.factor('prior_factor', prior_factor)
        else:
            # u = numpyro.sample(f"u", dist.Normal(0.0, 1.0).expand(tuple(pixelpop_data.bins)).to_event(pixelpop_data.dimension))
            # u = u - jnp.mean(u)
            # logZ = LSE(u)
            # phi = u - (logZ + jnp.sum(pixelpop_data.logdV))

            # numpyro.deterministic(f"phi", phi)
            # numpyro.deterministic(f"logZ", logZ + jnp.sum(pixelpop_data.logdV))
            # icar_dist = ICAR_model(single_dimension_adj_matrices=pixelpop_data.adj_matrices,
            #                        is_sparse=True,
            #                        dx=jnp.exp(jnp.sum(pixelpop_data.logdV)),
            #                        # validate_args=False,  # avoids the dependent-constraint validation issue
            #                       )
            # numpyro.factor(f"prior_factor", icar_dist.log_prob(phi))
            
            
            
            icar = ICAR_model(single_dimension_adj_matrices=pixelpop_data.adj_matrices, is_sparse=True, dx=jnp.exp(jnp.sum(pixelpop_data.logdV)))
            merger_rate_density = numpyro.sample('merger_rate_density', dist.ImproperUniform(dist.constraints.real, tuple(pixelpop_data.bins), ()))
            prior_factor, quad = icar.log_prob_and_quad(merger_rate_density)
            numpyro.factor('prior_factor', prior_factor)      
            #normalization = numpyro.deterministic('log_rate', LSE(merger_rate_density)+jnp.sum(pixelpop_data.logdV))
            
            # for ii, p in enumerate(pixelpop_data.pixelpop_parameters):
            #     sum_axes = tuple(np.arange(pixelpop_data.dimension)[np.r_[0:ii,ii+1:pixelpop_data.dimension]])
            #     numpyro.deterministic(f'log_marginal_{p}', LSE(merger_rate_density-normalization, axis=sum_axes) + jnp.sum(pixelpop_data.logdV[:ii]) + jnp.sum(pixelpop_data.logdV[ii+1:]))

        # unscaled_gamma = numpyro.sample('unscaled_gamma', numpyro.distributions.Gamma(concentration=(normalization_dof/2)))
        # SAL: As per Jack's comments, quad math might be wrong. Keep as-is for now. 
        # precision = unscaled_gamma * quad / 2
        # numpyro.deterministic('lnsigma', -0.5*jnp.log(precision))

        # SAL: We dont include the comoving factor here since it should multiply the entire mixture model
        event_weights_PP = merger_rate_density[event_bins] #phi[event_bins] 
        inj_weights_PP  = merger_rate_density[inj_bins] #phi[inj_bins]
        if log == 'debug':
            jaxprint('[DEBUG] pixelpop LSE(event_weights_PP)={ew}, LSE(inj_weights_PP)={iw}', ew=LSE(event_weights_PP), iw=LSE(inj_weights_PP))
        return event_weights_PP, inj_weights_PP

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
        
        # SAL: Calculate base event
        base_event = posteriors['ln_dVTc']-posteriors['log_prior']
        base_inj = injections['ln_dVTc']-injections['log_prior']

        # SAL: Calculate event, inj weights from PixelPop
        event_weights_PP, inj_weights_PP = nonparametric_model(
            pixelpop_data.event_bins, 
            pixelpop_data.inj_bins, 
            skip=pixelpop_data.skip_nonparametric
            )

        # SAL: Calculate event, inj weights from parametric models
        common_param_event_weights, common_param_inj_weights, strong_param_event_weights, strong_param_inj_weights = parametric_model(
            posteriors, 
            injections
            )

        # SAL: case in which we don't want pixelpop
        if pixelpop_data.skip_nonparametric:
            # jaxprint('Skipping non-parametric model')
            event_weights = base_event + event_weights_PP + common_param_event_weights + strong_param_event_weights
            inj_weights = base_inj + inj_weights_PP + common_param_inj_weights   + strong_param_inj_weights
        # SAL: case in which we only want pixelpop
        elif pixelpop_data.skip_mixture:
            # jaxprint('Skipping mixture model')
            log_rate_total = numpyro.sample('log_rate_total', dist.ImproperUniform(dist.constraints.real, (), ()))
            event_weights = base_event + log_rate_total + event_weights_PP + common_param_event_weights
            inj_weights = base_inj + log_rate_total + inj_weights_PP + common_param_inj_weights
        else:
            # jaxprint('Using mixture model')
            # SAL: implement mixture fraction
            log_rate_total = numpyro.sample('log_rate_total', dist.ImproperUniform(dist.constraints.real, (), ()))
            xi = numpyro.sample('xi', dist.Uniform(0., 1.)) # TODO: might be better a uniform distribution bc beta excludes 0 and 1
            # SAL: Implement mixture model (See notebook for details)
            # Note that base_event, base_inj corresponds to the log of the comoving factor dVc/dz (1/1+z)
            # Common param = models that factor out
            event_weights = base_event + log_rate_total + common_param_event_weights + jnp.logaddexp(jnp.log(xi) + event_weights_PP, jnp.log1p(-xi) + strong_param_event_weights)
            inj_weights = base_inj + log_rate_total + common_param_inj_weights + jnp.logaddexp(jnp.log(xi) + inj_weights_PP, jnp.log1p(-xi) + strong_param_inj_weights)

        ln_likelihood, nexp, pe_var, vt_var, total_var = \
            rate_likelihood(
                event_weights, 
                inj_weights, 
                injections['total_generated'], 
                live_time=injections['analysis_time']
                )
        taper = smooth(total_var, pixelpop_data.UncertaintyCut**2, 0.1) # "smooth" cutoff above Talbot+Golomb 2022 recommendation to retain autodifferentiability
        
        # save these values!
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