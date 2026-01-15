import numpy as np
from .gwpop_models import * 
from .car import initialize_ICAR, lower_triangular_log_prob, lower_triangular_map
from ..experimental.car import (
    DiagonalizedICARTransform, 
    initialize_sigma_marginalized_ICAR,
    lower_triangular_sigma_marg_log_prob
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
from numpyro.diagnostics import summary, print_summary
from jax import random
import os
from contextlib import redirect_stdout
import h5ify
from numpyro import handlers

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
            pixelpop_data.posteriors['ln_dVTc'].shape[0], 
            pixelpop_data.injections['ln_dVTc']-pixelpop_data.injections['log_prior'],
            random_initialization=pixelpop_data.random_initialization
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
            args, distribution = pixelpop_data.priors[key]
            if distribution.__name__ == 'Delta':
                sample[key] = args[0]
            else:
                sample[key] = numpyro.sample(key, distribution(*args))        
        
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

    if pixelpop_data.marginalize_sigma:
        ICAR_model = initialize_sigma_marginalized_ICAR(pixelpop_data.dimension)    
    else:
        ICAR_model = initialize_ICAR(pixelpop_data.dimension, length_scales=pixelpop_data.length_scales)

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

        if pixelpop_data.lower_triangular:
            
            base_interpolation = numpyro.sample('base_interpolation', dist.ImproperUniform(dist.constraints.real, unique_sample_shape, ()))
            merger_rate_density = numpyro.deterministic('merger_rate_density', lt_map(base_interpolation))
            
            if pixelpop_data.marginalize_sigma:
                prior_factor = lower_triangular_sigma_marg_log_prob(merger_rate_density, normalization_dof, pixelpop_data.adj_matrices)
            else:
                prior_factor = lower_triangular_log_prob(merger_rate_density, normalization_dof, lsigma, pixelpop_data.adj_matrices)
            numpyro.factor('prior_factor', prior_factor)

        else:

            if pixelpop_data.marginalize_sigma:
                merger_rate_density = numpyro.sample('merger_rate_density', ICAR_model(single_dimension_adj_matrices=pixelpop_data.adj_matrices, is_sparse=True))
            else:
                merger_rate_density = numpyro.sample('merger_rate_density', ICAR_model(log_sigmas=lsigma, single_dimension_adj_matrices=pixelpop_data.adj_matrices, is_sparse=True))        
            
            normalization = numpyro.deterministic('log_rate', LSE(merger_rate_density)+jnp.sum(pixelpop_data.logdV))
            for ii, p in enumerate(pixelpop_data.pixelpop_parameters):
                sum_axes = tuple(np.arange(pixelpop_data.dimension)[np.r_[0:ii,ii+1:pixelpop_data.dimension]])
                numpyro.deterministic(f'log_marginal_{p}', LSE(merger_rate_density-normalization, axis=sum_axes) + jnp.sum(pixelpop_data.logdV[:ii]) + jnp.sum(pixelpop_data.logdV[ii+1:]))

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
        event_weights, inj_weights = nonparametric_model(
            pixelpop_data.event_bins, 
            pixelpop_data.inj_bins, 
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

def get_worst_rhat_neff(chain_samples, skip_keys=[]):
    """
    Identify the parameter with the worst R-hat and effective sample size (Neff).

    Parameters
    ----------
    chain_samples : dict
        Dictionary of chain samples from NumPyro MCMC, with parameter name keys
    skip_keys : list
        List of keys to skip over in calculation of worst Rhat and Neff

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
        k = f'{rh[0]}{[int(p) for p in ind]}'.replace('[]','')
        if k not in skip_keys:
            name.append(rh[0])
            pos.append(ind)
            rhat_values.append(rh[1][ind])
    
    worst_rhat = np.argmax(rhat_values)
    rhat_chain = chain_samples[name[worst_rhat]][...,*pos[worst_rhat]]
    rhat_key = f'{name[worst_rhat]}{[int(p) for p in pos[worst_rhat]]}'.replace('[]','')

    name, pos, neff_values = [], [], []
    for rh in neffs:
        ind = np.unravel_index(np.argmin(rh[1], axis=None), rh[1].shape)
        k = f'{rh[0]}{[int(p) for p in ind]}'.replace('[]','')
        if k not in skip_keys:
            name.append(rh[0])
            pos.append(ind)
            neff_values.append(rh[1][ind])
        
    worst_neff = np.argmin(neff_values)
    neff_chain = chain_samples[name[worst_neff]][...,*pos[worst_neff]]
    neff_key = f'{name[worst_neff]}{[int(p) for p in pos[worst_neff]]}'.replace('[]','')
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
        if name.startswith('~'):
            continue
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
    skip_keys = [k[1:] for k in print_keys if k.startswith('~')]
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
        sample_iterator = tqdm(range(int(1e-4 + tot_samples/num_samples)))
        sample_iterator.set_description("drawing thinned samples")
        for sample in sample_iterator:
            mcmc.post_warmup_state = mcmc.last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, **model_kwargs)
            next_sample = mcmc.get_samples()
            sys.stdout.write("\x1b[1A\n\x1b[1A")

            if chain_samples is None:
                chain_samples = {key: np.array(next_sample[key]) for key in next_sample}
            else:
                for key in chain_samples:
                    chain_samples[key] = np.concatenate((chain_samples[key], np.array(next_sample[key])), axis=0)
            mcmc.transfer_states_to_host()
            key0 = list(chain_samples.keys())[0]
            if (sample % cache_cadence == 0) and (chain_samples[key0].shape[0] >= 4):
                sys.stdout.write(f"\x1b[1A\x1b[2K"*(table_size+3)) # move the cursor up to overwrite the summary table for the NEXT print
                
                rhat, rhat_chain, neff, neff_chain = get_worst_rhat_neff(chain_samples, skip_keys=skip_keys)
                summary_dict = {key: chain_samples[key] for key in print_keys if key[1:] not in skip_keys}
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