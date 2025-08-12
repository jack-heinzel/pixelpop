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
import os
from contextlib import redirect_stdout
import h5ify

def setup_probabilistic_model(
        posteriors, injections, parameters, other_parameters, bins, length_scales=False, 
        minima={}, maxima={}, parametric_models={}, hyperparameters={}, priors={}, 
        plausible_hyperparameters={}, UncertaintyCut=1., random_initialization=False, 
        lower_triangular=False, prior_draw=False, skip_nonparametric=False, 
        constraint_funcs=[], log='default'
        ):
    '''
    Parameters
    ----------
    posteriors: dict
        A dictionary of the posterior samples, gwparameter key (e.g., mass_1)
        to (j)np.NDarray shaped as (Nobs,Nsample). Should also contain a 
        'ln_prior' entry
    injections: dict
        A dictionary of the found injections, gwparameter key (e.g., mass_1)
        to (j)np.NDarray shaped as (Nfound). Should also contain a 'ln_prior'
        entry, and 'total_generated': float/int and 'analysis_time': float
        entry
    parameters: list
        list of strings, containing the parameters over which the pixelpop model
        was defined    
    other_parameters: list
        list of strings, containing the parameters for the other parameters 
        necessary in the population model
    bins: int or list
        number of bins along each axis
    length_scales: bool
        whether to assume a different coupling parameter along each direction.
        By default is set to false, e.g., uses the same universal coupling 
        between nearest neighbor bins
    minima: dict
        dictionary of gwparameter keys (mass_1, mass_ratio, chi_eff, etc.) to the 
        minimum value in the space. If no key is passed, defaults to typical bbh 
        values, e.g., mass_1: 3, mass_ratio: 0., chi_eff: -1
    maxima: dict
        dictionary of gwparameter keys (mass_1, mass_ratio, chi_eff, etc.) to the 
        maximum value in the space. If no key is passed, defaults to typical bbh 
        values, e.g., mass_1: 200, mass_ratio: 1., chi_eff: 1
    parametric_models: dict    
        dictionary of gwparameter keys (mass_1, mass_ratio, chi_eff, etc.) to 
        parametric model function. If no key is passed, defaults to GWTC-3 default 
        parametric models
    hyperparameters: dict
        dictionary of gwparameter keys (mass_1, mass_ratio, chi_eff, etc.) to a list 
        of the (string) hyperparameter names for the corresponding parametric 
        function
    priors: dict
        dictionary of hyperparmeter name to a tuple containing (zeroth index) the 
        list of arguments for the numpyro distribution from which to sample the 
        hyperparameter, and (first index) the numpyro distribution, e.g., 
        'max_z': ([2.4], numpyro.distributions.Delta)
    plausible_hyperparameters: dict
        dictionary of gwparameter keys (mass_1, mass_ratio, chi_eff, etc.) to a 
        plausible value of the parameter. Used for initializing the pixelpop model
        to a reasonable place.
    UncertaintyCut: float
        value above which the likelihood uncertainty is regarded as too large, and 
        is regularized sharply
    random_initialization: bool
        whether to initialize to a "reasonable" pixelpop model or a random white noise
        for warmup
    lower_triangular: bool
        whether to use the lower_triangular formalism where p1 > p2 is assumed.
        Usually this will only be used when joint mass_1, mass_2 inference is done 
    prior_draw: bool
        whether to include the likelihood in the probabilistic model
    skip_nonparametric: bool
        whether to only sample the parametric models
    constraint_funcs: list
        extra constraints on the hyperparameter prior spaces
    log: str
        Either "default" or "debug". In debug mode, print more values along the way
        for debugging
    
    Returns
    -------
    probabilistic_model: function
        numpyro probabilistic model for use in numpyro MCMC methods
    initial_value: dict
        dictionary of initial values for starting warmup of MCMC
    '''
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
        normalization_dof = tri_size * int(np.prod(bins[2:])) # lower triangular in first two dimensions

    def get_initial_value(plausible_hyperparameters, parameters, Nobs, inj_weights, random_initialization):
        bin_med = [(bin_axes[ii][:-1] + bin_axes[ii][1:])/2 for ii in range(dimension)]
        # print(bin_med)
        interpolation_grid = np.meshgrid(*bin_med, indexing='ij')

        if random_initialization:
            if lower_triangular:
                return {'base_interpolation': np.random.normal(loc=0, scale=2, size=unique_sample_shape)}
            else:
                return {'merger_rate_density': np.random.normal(loc=0, scale=2, size=interpolation_grid[0].shape)}
        else:
            data_grid = {p.replace('_psi',''): interpolation_grid[ii] for ii, p in enumerate(parameters)}    
            
            initial_interpolation = np.sum([
                parameter_to_gwpop_model[p](data_grid, *[plausible_hyperparameters[h] for h in parameter_to_hyperparameters[p]]) for ii, p in enumerate(parameters)
            ], axis=0)
            pdet = LSE(initial_interpolation[inj_bins] + inj_weights) - jnp.log(injections['total_generated'])
            Rexp = jnp.log(Nobs) - pdet - jnp.log(injections['analysis_time'])
            initial_interpolation = np.logaddexp(initial_interpolation, -10*np.ones_like(initial_interpolation)) # logaddexp -10 to smooth out negative divergences
            return {'merger_rate_density': Rexp + initial_interpolation}

    parameters_psi = [p.replace('redshift', 'redshift_psi') for p in parameters]
    if skip_nonparametric:
        initial_value = {}
    else:
        initial_value = get_initial_value(hyperparameters_plausible, parameters_psi, event_ln_dVTc.shape[0], inj_ln_dVTc-injections['log_prior'], random_initialization=random_initialization)

    def parametric_model(data, injections, event_weights, inj_weights):
        sample = {}
        for key in hyperparameter_priors:
            args, distribution = hyperparameter_priors[key]
            if distribution.__name__ == 'Delta':
                sample[key] = args[0]
            else:
                sample[key] = numpyro.sample(key, distribution(*args))        
        if log == 'debug':
            for p in other_parameters:
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

    ICAR_model = initialize_ICAR(dimension, length_scales=length_scales)
    def nonparametric_model(event_bins, inj_bins, event_weights, inj_weights, skip=False):
        if skip:
            R = numpyro.sample('log_rate', dist.ImproperUniform(dist.constraints.real, (), ()))
            return event_weights + R[None,None], inj_weights + R[None]
        if length_scales:
            lsigma = numpyro.sample('lnsigma', dist.Uniform(-3,3), sample_shape=(dimension,))
        else:
            lsigma = numpyro.sample('lnsigma', dist.Uniform(-3,3), sample_shape=()) 

        if lower_triangular:
            base_interpolation = numpyro.sample('base_interpolation', dist.ImproperUniform(dist.constraints.real, unique_sample_shape, ()))
            merger_rate_density = numpyro.deterministic('merger_rate_density', lt_map(base_interpolation))
            numpyro.factor('prior_factor', lower_triangular_log_prob(merger_rate_density, normalization_dof, lsigma, adj_matrices))

        else:
            merger_rate_density = numpyro.sample('merger_rate_density', ICAR_model(log_sigmas=lsigma, single_dimension_adj_matrices=adj_matrices, is_sparse=True))        
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
        
        event_weights, inj_weights = nonparametric_model(event_bins, inj_bins, event_ln_dVTc-posteriors['log_prior'], inj_ln_dVTc-injections['log_prior'], skip=skip_nonparametric)
        event_weights, inj_weights = parametric_model(posteriors, injections, event_weights, inj_weights)

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

def inference_loop(
    probabilistic_model, model_kwargs={}, initial_value={}, warmup=10000, tot_samples=100, thinning=100, pacc=0.65, maxtreedepth=10, 
    num_samples=1, parallel=1, rng_key=random.PRNGKey(1), cache_cadence=1, run_dir='./', name='',
    print_keys=['Nexp', 'log_likelihood', 'log_likelihood_variance'], dense_mass=False, chain_offset=0
    ):
    '''
    Parameters
    ----------
    probabilistic_model: function
        numpyro probabilistic model for use in numpyro MCMC methods
    model_kwargs: dict
        dictionary of kwargs for use in probabilistic model, usually will be 
        'posteriors': dict of gwposterior, 'injections': dict of found injections
    initial_value: dict
        dictionary of initial value to initialize to before warmup phase
    warmup: int
        number of warmup iterations
    tot_samples: int
        total number of posterior samples to save per chain
    thinning: int
        number of steps between which each sample is recorded
    pacc: float
        target acceptance probability for NUTS sampler
    maxtreedepth: int
        take no more than 2^maxtreedepth - 1 steps in a particular iteration
    num_samples: int
        number of samples between which the chain metadata is printed (e.g., Neff, 
        rhats, means of hyperparameters)
    parallel: int
        number of independent sequential chains to sample
    rng_key: `jax.random.PRNGKey` object
    cache_cadence: int
        number of samples between which the chain is saved
    run_dir: str
        path to directory where to save the posterior samples
    name: str
        name of directory where to save the posterior samples, if this doesn't 
        exist, create a new one with this name
    print_keys: list
        list of hyperparameter names to always print in intermediate numpyro summary
        tables, e.g., ['Nexp', 'log_likelihood_variance'], etc.
    dense_mass: bool
        whether to optimize a mass matrix including off-diagonal terms. Definitely you
        should basically always keep this off.
    chain_offset: int
        when saving new chains, offset the chain_num in the name 
        f'chain_{chain_num}_samples.h5' by this offset. Useful whenever there are 
        existing chains that you want to keep. Otherwise, chains are silently over-
        written.

    Returns
    -------
    samples: list
        list of dictionaries of samples in independent chains
    mcmc: `numpyro.infer.mcmc.MCMC` object
        completed MCMC sampler (probably we don't need to return this)
    '''

    kernel = NUTS(probabilistic_model, max_tree_depth=maxtreedepth, target_accept_prob=pacc, init_strategy=numpyro.infer.init_to_value(values=initial_value), dense_mass=dense_mass)

    samples = []
    rng_keys = random.split(rng_key, num=parallel)
    for chain in range(parallel):
        rng_key = rng_keys[chain]
        print(f"Warming up chain #{chain + 1} out of {parallel}")
        mcmc = MCMC(kernel, thinning=thinning, num_warmup=warmup, num_samples=num_samples*thinning, num_chains=1)# , chain_method='vectorized')# , chain_method='sequential') # vectorized is an experimental method. We can pass 'parallel' which attempts to distribute the chains across multiple GPUs, e.g. on pcdev12 we could do num_chains = 4 across the a100s. If num_chains is too large, it defaults to 'sequential' which simply evaluates the chains in series.
        
        mcmc.warmup(rng_key, **model_kwargs)
        table_size = len(print_keys) + 2
        sys.stdout.write("\n"*(table_size+3)) # buffer line between the progress bars
        chain_samples = None
        mcmc.transfer_states_to_host()
        sample_iterator = tqdm(range(int(1e-4 + tot_samples/num_samples)))
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