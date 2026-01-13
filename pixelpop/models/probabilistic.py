import numpy as np
from ..utils.nearest_neighbor import create_CAR_coupling_matrix
from .gwpop_models import * 
from .car import initialize_ICAR, lower_triangular_log_prob, lower_triangular_map
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.debug import print as jaxprint
from ..utils.data import place_in_bins
from jax.scipy.special import logsumexp as LSE
import numpyro
from numpyro.infer import MCMC, NUTS
from tqdm import tqdm
import sys
from numpyro.diagnostics import summary, print_summary
from jax import random
from jax.nn import sigmoid
import os
from contextlib import redirect_stdout
import h5ify
from numpyro import handlers
from ..experimental.car import (
    initialize_sigma_marginalized_ICAR,
    lower_triangular_sigma_marg_log_prob
)

def setup_probabilistic_model(
    posteriors, 
    injections, 
    parameters, 
    other_parameters, 
    bins, 
    length_scales=False, 
    minima={}, 
    maxima={}, 
    parametric_models={}, 
    hyperparameters={}, 
    priors={}, 
    plausible_hyperparameters={}, 
    UncertaintyCut=1., 
    random_initialization=True, 
    lower_triangular=False, 
    prior_draw=False, 
    skip_nonparametric=False, 
    constraint_funcs=[], 
    log='default', 
    scale_by_sigma=None, 
    coupling_prior=[(-3,3), dist.Uniform],
    sample_eigenbasis=False, 
    marginalize_sigma=False,
):
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
    length_scales : bool, optional
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
    plausible_hyperparameters : dict, optional
        Mapping of parameter → plausible hyperparameter values (for initialization).
    UncertaintyCut : float, optional
        Cutoff for regularizing large likelihood uncertainties (default 1.0).
    random_initialization : bool, optional
        If True, initialize ICAR model with random noise instead of plausible values.
    lower_triangular : bool, optional
        If True, enforce p1 > p2 triangular support (used for joint m1–m2 models).
    prior_draw : bool, optional
        If True, construct model without likelihood factor.
    skip_nonparametric : bool, optional
        If True, disable the pixelized (nonparametric) component.
    constraint_funcs : list of callables, optional
        Extra constraint functions applied to hyperparameters.
    log : {"default", "debug"}, optional
        Logging verbosity.
    scale_by_sigma : None or bool
        If true, sample from CAR model with sigma = 1 and scale by sigma. If None,
        defaults to prior_draw boolean value

    Returns
    -------
    probabilistic_model : callable
        NumPyro-compatible probabilistic model.
    initial_value : dict
        Suggested initial values for MCMC warmup.
    """
    if scale_by_sigma is None:
        scale_by_sigma = prior_draw
    if scale_by_sigma and length_scales:
        raise ValueError("Cannot scale by different sigmas in different axes")
    if marginalize_sigma and length_scales:
        raise ValueError("Cannot marginalize over sigma with different sigmas in different axes") 

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
        
    event_bins, inj_bins, bin_axes, logdV = place_in_bins(parameters, posteriors, injections, bins=bins, minima=minima, maxima=maxima)
    
    # update models
    parameter_to_hyperparameters = gwparameter_to_hyperparameters.copy()
    parameter_to_hyperparameters.update(hyperparameters)

    hyperparameters_plausible = typical_hyperparameters.copy()
    hyperparameters_plausible.update(plausible_hyperparameters)

    parameter_to_gwpop_model = {}
    # S: update all possible models, including those of PP parameters
    for p in parameters+other_parameters:
        if p in parametric_models:
            print(f'Updating {p} model from {gwparameter_to_model[p].__name__} to {parametric_models[p].__name__}')
            print(f'\t ...with hyperparameters {parameter_to_hyperparameters[p]}')
            parameter_to_gwpop_model[p] = parametric_models[p]
        else:
            print(f'Using default {p} model {gwparameter_to_model[p].__name__}')
            parameter_to_gwpop_model[p] = gwparameter_to_model[p]

    # default priors
    hyperparameter_priors = {}
    # S: update all possible models, including those of PP parameters
    for p in parameters+other_parameters:
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
        normalization_dof = tri_size * int(np.prod(bins[2:])) # lower triangular in first two dimensions

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
        bin_med = [(bin_axes[ii][:-1] + bin_axes[ii][1:])/2 for ii in range(dimension)]
        # print(bin_med)
        interpolation_grid = np.meshgrid(*bin_med, indexing='ij')
        if scale_by_sigma:
            return_key = 'latent_density'
        else:
            return_key = 'merger_rate_density'
        if random_initialization:
            if lower_triangular:
                return {'base_interpolation': np.random.normal(loc=0, scale=2, size=unique_sample_shape)}
            else:
                return {return_key: np.random.normal(loc=0, scale=2, size=interpolation_grid[0].shape)}
        else:
            data_grid = {p.replace('_psi',''): interpolation_grid[ii] for ii, p in enumerate(parameters)}    
            
            initial_interpolation = np.sum([
                parameter_to_gwpop_model[p](data_grid, *[plausible_hyperparameters[h] for h in parameter_to_hyperparameters[p]]) for ii, p in enumerate(parameters)
            ], axis=0)
            pdet = LSE(initial_interpolation[inj_bins] + inj_weights) - jnp.log(injections['total_generated'])
            Rexp = jnp.log(Nobs) - pdet - jnp.log(injections['analysis_time'])
            initial_interpolation = np.logaddexp(initial_interpolation, -10*np.ones_like(initial_interpolation)) # logaddexp -10 to smooth out negative divergences
            return {return_key: Rexp + initial_interpolation}    
            
    parameters_psi = [p.replace('redshift', 'redshift_psi') for p in parameters]
    if skip_nonparametric:
        initial_value = {}
    else:
        initial_value = get_initial_value(hyperparameters_plausible, parameters_psi, event_ln_dVTc.shape[0], inj_ln_dVTc-injections['log_prior'], random_initialization=random_initialization)

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
        common_param_event = 0.
        common_param_inj = 0.
        strong_param_event = 0.
        strong_param_inj = 0.
        
        for key in hyperparameter_priors:
            args, distribution = hyperparameter_priors[key]
            if distribution.__name__ == 'Delta':
                sample[key] = args[0]
            else:
                sample[key] = numpyro.sample(key, distribution(*args))        
        if log == 'debug':
            # S: need to debug all parameters
            for p in parameters+other_parameters:
                jaxprint('[DEBUG] =================================')
                jaxprint('[DEBUG] parametric parameters: {p}', p=p)
                jaxprint('[DEBUG] =================================')       
                for k in parameter_to_hyperparameters[p]:
                    jaxprint('[DEBUG] \t {k} sample: {s}', k=k, s=sample[k])
                    
        for constraint_func in constraint_funcs:
            numpyro.factor(constraint_func.__name__, constraint_func(sample))
            if log == 'debug':
                jaxprint('[DEBUG] constraint functions:', constraint_func.__name__, constraint_func(sample))
                
        for p in other_parameters:
            common_param_event += parameter_to_gwpop_model[p](data, *[sample[h] for h in parameter_to_hyperparameters[p]])
            common_param_inj += parameter_to_gwpop_model[p](injections, *[sample[h] for h in parameter_to_hyperparameters[p]])
            if log == 'debug':
                jaxprint('[DEBUG] parametric {p} LSE(event_weights)={ew}, LSE(injection_weights)={iw}', p=p, ew=LSE(common_param_event), iw=LSE(common_param_inj))
                if not jnp.isfinite(LSE(common_param_event)):
                    for parameter in map_to_gwpop_parameters[p]:
                        jaxprint('[DEBUG] inf event weights at {p}={d}', p=parameter, d=data[parameter][jnp.where(common_param_event == jnp.inf)])
                if not jnp.isfinite(LSE(common_param_inj)):
                    for parameter in map_to_gwpop_parameters[p]:
                        jaxprint('[DEBUG] inf injection weights at {p}={d}', p=parameter, d=injections[parameter][jnp.where(common_param_inj == jnp.inf)])
        
        for sp in parameters:
            strong_param_event += parameter_to_gwpop_model[sp](data, *[sample[h] for h in parameter_to_hyperparameters[sp]])
            strong_param_inj += parameter_to_gwpop_model[sp](injections, *[sample[h] for h in parameter_to_hyperparameters[sp]])
            if log == 'debug':
                jaxprint('[DEBUG] strong {p} LSE(event_weights)={ew}, LSE(injection_weights)={iw}', p=sp, ew=LSE(strong_param_event), iw=LSE(strong_param_inj))
                if not jnp.isfinite(LSE(strong_param_event)):
                    for parameter in map_to_gwpop_parameters[sp]:
                        jaxprint('[DEBUG] inf event weights at {p}={d}', p=parameter, d=data[parameter][jnp.where(strong_param_event == jnp.inf)])
                if not jnp.isfinite(LSE(strong_param_inj)):
                    for parameter in map_to_gwpop_parameters[sp]:
                        jaxprint('[DEBUG] inf injection weights at {p}={d}', p=parameter, d=injections[parameter][jnp.where(strong_param_inj == jnp.inf)])
                
        # S: sample log_R_strong
        log_R_strong = numpyro.sample('log_R_strong', dist.Uniform(-10.0, 10.0))  
        
        return common_param_event, common_param_inj, strong_param_event, strong_param_inj, log_R_strong

    if marginalize_sigma:
        ICAR_model = initialize_sigma_marginalized_ICAR(dimension)    
    else:
        ICAR_model = initialize_ICAR(dimension, length_scales=length_scales)
    
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
        event_weights : ndarray
            Updated event log-weights including nonparametric contributions.
        inj_weights : ndarray
            Updated injection log-weights including nonparametric contributions.
        """
        
        if skip:
            R = numpyro.sample('log_rate', dist.ImproperUniform(dist.constraints.real, (), ()))
            # S: event, inj
            return R[None,None], R[None]
            
        if length_scales:
            lsigma = numpyro.sample('lnsigma', coupling_prior[1](*coupling_prior[0]), sample_shape=(dimension,))
        else:
            lsigma = numpyro.sample('lnsigma', coupling_prior[1](*coupling_prior[0]), sample_shape=()) 

        if lower_triangular:
            base_interpolation = numpyro.sample('base_interpolation', dist.ImproperUniform(dist.constraints.real, unique_sample_shape, ()))
            
            if scale_by_sigma:
                merger_rate_density = numpyro.deterministic('merger_rate_density', lt_map(base_interpolation*jnp.exp(lsigma)))
                numpyro.factor('prior_factor', lower_triangular_log_prob(merger_rate_density, normalization_dof, 0., adj_matrices))
            else:
                merger_rate_density = numpyro.deterministic('merger_rate_density', lt_map(base_interpolation))
                numpyro.factor('prior_factor', lower_triangular_log_prob(merger_rate_density, normalization_dof, lsigma, adj_matrices))   

            # S: need to calculate normalization but divide by 2 so we don't add twice.
            normalization = numpyro.deterministic('log_rate', LSE(merger_rate_density_tri) + jnp.sum(logdV) - jnp.log(2))
            
            # S: add this line for mixture model diagnostics
            log_R_pixel = numpyro.deterministic('log_R_pixel', normalization) 
        
        else:
            
            if scale_by_sigma:
                latent_density = numpyro.sample('latent_density', ICAR_model(log_sigmas=0., single_dimension_adj_matrices=adj_matrices, is_sparse=True))
                merger_rate_density = numpyro.deterministic('merger_rate_density', jnp.exp(lsigma) * latent_density) 
            else:
                merger_rate_density = numpyro.sample('merger_rate_density', ICAR_model(log_sigmas=lsigma, single_dimension_adj_matrices=adj_matrices, is_sparse=True))   
            normalization = numpyro.deterministic('log_rate', LSE(merger_rate_density)+jnp.sum(logdV))
            # S: add this line for mixture model diagnostics
            log_R_pixel = numpyro.deterministic('log_R_pixel', normalization) 
            for ii, p in enumerate(parameters):
                sum_axes = tuple(np.arange(dimension)[np.r_[0:ii,ii+1:dimension]])
                numpyro.deterministic(f'log_marginal_{p}', LSE(merger_rate_density-normalization, axis=sum_axes) + jnp.sum(logdV[:ii]) + jnp.sum(logdV[ii+1:]))

        event_weights_PP = merger_rate_density[event_bins] # (69,3194)
        inj_weights_PP = merger_rate_density[inj_bins]
        if log == 'debug':
            jaxprint('[DEBUG] pixelpop LSE(event_weights)={ew}, LSE(injection_weights)={iw}', ew=LSE(event_weights_PP), iw=LSE(inj_weights_PP))
        if prior_draw:
            numpyro.factor("effective_likelihood", -jnp.mean(merger_rate_density**2) / 2 / jnp.exp(2*lsigma))
        return event_weights_PP, inj_weights_PP, log_R_pixel

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
        # S: calculate base event
        base_event = event_ln_dVTc-posteriors['log_prior']
        base_inj = inj_ln_dVTc-injections['log_prior']

        # S: calculate event, inj weights from PixelPop
        event_weights_PP, inj_weights_PP, log_R_pixel = nonparametric_model(event_bins, inj_bins, skip=skip_nonparametric)
        common_param_event, common_param_inj, strong_param_event, strong_param_inj, log_R_strong = parametric_model(posteriors, injections)

        # S: Mixture model. See notebook for details
        event_weights = base_event + common_param_event + jnp.logaddexp(event_weights_PP, log_R_strong + strong_param_event)
        inj_weights = base_inj + common_param_inj +  jnp.logaddexp(inj_weights_PP, log_R_strong + strong_param_inj)

        # S: Diagnostics for the mixture model
        numpyro.deterministic('xi_rate', sigmoid(log_R_pixel - log_R_strong))

        ln_likelihood, nexp, pe_var, vt_var, total_var = rate_likelihood(event_weights, inj_weights, injections['total_generated'], live_time=injections['analysis_time'])
        taper = smooth(total_var, UncertaintyCut**2, 0.1) # "smooth" cutoff above Talbot+Golomb 2022 recommendation to retain autodifferentiability
        
        # save these values!
        numpyro.deterministic("log_likelihood", ln_likelihood)
        numpyro.deterministic("log_likelihood_variance", total_var)
        numpyro.deterministic("pe_variance", pe_var)
        numpyro.deterministic("vt_variance", vt_var)
        numpyro.deterministic("Nexp", nexp)

        if not prior_draw:
            numpyro.factor("log_likelihood_plus_taper", ln_likelihood + taper)

    return probabilistic_model, initial_value


def get_worst_rhat_neff(chain_samples):
    """
    Identify the parameter with the worst R-hat and effective sample size (Neff).

    Parameters
    ----------
    chain_samples : dict
        Dictionary of chain samples from NumPyro MCMC, with parameter name keys

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
    chain_summary = summary(chain_samples, group_by_chain=False)
    rhats = [[key, chain_summary[key]['r_hat']] for key in chain_summary]
    neffs = [[key, chain_summary[key]['n_eff']] for key in chain_summary]
    
    name, pos, rhat_values = [], [], []
    for rh in rhats:
        ind = np.unravel_index(np.argmax(rh[1], axis=None), rh[1].shape)
        name.append(rh[0])
        pos.append(ind)
        rhat_values.append(rh[1][ind])
    
    worst_rhat = np.argmax(rhat_values)
    rhat_chain = chain_samples[name[worst_rhat]][...,*pos[worst_rhat]]
    rhat_key = f'{name[worst_rhat]}{list(pos[worst_rhat])}'.replace('[]','')

    name, pos, neff_values = [], [], []
    for rh in neffs:
        ind = np.unravel_index(np.argmin(rh[1], axis=None), rh[1].shape)
        name.append(rh[0])
        pos.append(ind)
        neff_values.append(rh[1][ind])
    
    worst_neff = np.argmin(neff_values)
    neff_chain = chain_samples[name[worst_neff]][...,*pos[worst_neff]]
    neff_key = f'{name[worst_neff]}{list(pos[worst_neff])}'.replace('[]','')
    return rhat_key, rhat_chain, neff_key, neff_chain

def get_table_size(probabilistic_model, initial_value, model_kwargs, print_keys):
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

    Returns
    -------
    size : int
        Number of rows expected in the summary table.
    """
    conditioned_model = handlers.condition(probabilistic_model, data=initial_value)
    with handlers.seed(rng_seed=0):
        trace = handlers.trace(conditioned_model).get_trace(**model_kwargs)

    size = 2
    for name in print_keys:
        try:
            size += trace[name]["value"].size
        except KeyError:
            raise KeyError(f'You are trying to print \"{name}\", valid print_keys are {list(trace.keys())}')
    return size

def inference_loop(
    probabilistic_model, model_kwargs={}, initial_value={}, warmup=10000, tot_samples=100, thinning=100, pacc=0.65, maxtreedepth=10, 
    num_samples=1, parallel=1, rng_key=random.PRNGKey(1), cache_cadence=1, run_dir='./', name='',
    print_keys=['Nexp', 'log_likelihood', 'log_likelihood_variance'], dense_mass=False, chain_offset=0
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
    parallel : int, optional
        Number of independent chains to run (default 1).
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
    dense_mass : bool, optional
        Whether to use a dense mass matrix in NUTS (default False).
    chain_offset : int, optional
        Offset applied to chain index when saving outputs (default 0).

    Returns
    -------
    samples : list of dict
        List of posterior samples for each chain.
    mcmc : numpyro.infer.MCMC
        Completed MCMC sampler instance.
    """

    table_size = get_table_size(probabilistic_model, initial_value, model_kwargs, print_keys)

    kernel = NUTS(probabilistic_model, max_tree_depth=maxtreedepth, target_accept_prob=pacc, init_strategy=numpyro.infer.init_to_value(values=initial_value), dense_mass=dense_mass)

    samples = []
    rng_keys = random.split(rng_key, num=parallel)
    for chain in range(parallel):
        rng_key = rng_keys[chain]
        print(f"Warming up chain #{chain + 1} out of {parallel}")
        mcmc = MCMC(kernel, thinning=thinning, num_warmup=warmup, num_samples=num_samples*thinning, num_chains=1)# , chain_method='vectorized')# , chain_method='sequential') # vectorized is an experimental method. We can pass 'parallel' which attempts to distribute the chains across multiple GPUs, e.g. on pcdev12 we could do num_chains = 4 across the a100s. If num_chains is too large, it defaults to 'sequential' which simply evaluates the chains in series.
        
        mcmc.warmup(rng_key, **model_kwargs)
        sys.stdout.write("\n"*(table_size+3)) # buffer line between the progress bars
        chain_samples = None
        mcmc.transfer_states_to_host()
        sample_iterator = tqdm(range(int(1e-4 + tot_samples/num_samples)-1))
        sample_iterator.set_description("drawing thinned samples")
        for sample in sample_iterator:
            mcmc.post_warmup_state = mcmc.last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, **model_kwargs)
            next_sample = mcmc.get_samples()
            sys.stdout.write("\x1b[1A\n\x1b[1A")

            if chain_samples is None:
                chain_samples = {key:np.array(next_sample[key]) for key in next_sample}
            for key in chain_samples:
                chain_samples[key] = np.concatenate((chain_samples[key], np.array(next_sample[key])), axis=0)
            mcmc.transfer_states_to_host()
            if (sample % cache_cadence == 0) and (chain_samples[key].shape[0] >= 4):
                sys.stdout.write(f"\x1b[1A\x1b[2K"*(table_size+3)) # move the cursor up to overwrite the summary table for the NEXT print
                
                rhat, rhat_chain, neff, neff_chain = get_worst_rhat_neff(chain_samples)
                summary_dict = {key: chain_samples[key] for key in print_keys}
                summary_dict['worst r_hat: '+rhat] = rhat_chain
                summary_dict['worst n_eff: '+neff] = neff_chain
                
                print_summary(summary_dict, group_by_chain=False)

                os.makedirs(os.path.join(run_dir, name), exist_ok=True)
                with open(os.path.join(run_dir, name, f'chain_{chain+chain_offset}_metadata.txt'), 'w+') as f:
                    with redirect_stdout(f):
                        print_summary(summary_dict, group_by_chain=False)
                f = os.path.join(run_dir, name, f'chain_{chain+chain_offset}_samples.h5')
                h5ify.save(f, chain_samples, mode='w')
        
        samples.append(chain_samples)

    return samples, mcmc