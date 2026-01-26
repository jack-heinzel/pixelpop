import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as LSE
from tqdm import tqdm
import numpy as np
import warnings

class PixelPopRateFunction(object):
    """
    A wrapper class that converts PixelPop data and model settings into a 
    callable rate function compatible with the `population_error` package.

    This class combines the parametric components (e.g., for "nuisance" parameters)
    and the non-parametric pixelized components of the PixelPop model to return
    the expected merger rate density for a given set of hyperparameters.

    Parameters
    ----------
    pixelpop_data : PixelPopData
        The data container holding event posteriors, injections, bin definitions,
        and model settings.
    dataset_type : str, default='posteriors'
        Specifies which dataset bins to use for the rate evaluation. 
        Must be either 'posteriors' (for evaluating event rates) or 
        'injections' (for evaluating selection sensitivity).

    Attributes
    ----------
    dataset_bins : jax.numpy.ndarray
        The pre-computed bin indices for the specified dataset.
    other_parameters : list of str
        List of parameters modeled by parametric functions rather than pixels.
    parametric_models : dict
        Dictionary mapping parameters to their corresponding model functions.
    parameter_to_hyperparameters : dict
        Dictionary mapping parameters to the list of required hyperparameters.

    Methods
    -------
    __call__(dataset, hyperparameters)
        Computes the rate density for the provided dataset and hyperparameters.
    """
    
    def __init__(self, pixelpop_data, dataset_type='posteriors'):

        attrs_to_copy = [
            'other_parameters', 
            'parameter_to_hyperparameters', 
            'parametric_models', 
            ]
        
        for attr in attrs_to_copy:
            value = getattr(pixelpop_data, attr)
            setattr(self, attr, value)

        if dataset_type == 'posteriors':
            self.dataset_bins = pixelpop_data.event_bins
        elif dataset_type == 'injections':
            self.dataset_bins = pixelpop_data.inj_bins
        else:
            raise ValueError(
                f'dataset_type can only be \'posteriors\' or \'injections\', you entered {dataset_type}' 
                )
        self.shape = self.dataset_bins[0].shape

    def __call__(self, dataset, hyperparameters):
        """
        Evaluate the merger rate density at the dataset points.

        Parameters
        ----------
        dataset : dict
            Dictionary containing the data samples (e.g., 'ln_dVTc'). 
            Note: The actual bin locations are pre-stored in `self.dataset_bins` 
            and are not extracted from this dictionary during the call.
        hyperparameters : dict
            Dictionary of population hyperparameters, including 'merger_rate_density'
            (the pixel heights) and parameters for any parametric sub-models.

        Returns
        -------
        jax.numpy.ndarray
            The expected rate density (in units of probability * total rate) 
            for each sample in the dataset.
        """
        lp_parametric = self.log_prob_parametric_model(dataset, hyperparameters)
        lp_pixelpop = self.log_rate_pixelpop(dataset, hyperparameters)

        return jnp.exp(lp_parametric + lp_pixelpop)
    
    def log_prob_parametric_model(self, dataset, hyperparameters):
        
        log_probs = jnp.zeros(self.shape)
              
        for p in self.other_parameters:
            hs = self.parameter_to_hyperparameters[p] # hyperparameters appropriate for this model
            log_probs += self.parametric_models[p](dataset, *[hyperparameters[h] for h in hs])
            
        return log_probs
    
    def log_rate_pixelpop(self, dataset, hyperparameters):

        ln_dVTc = dataset['ln_dVTc']
        pp_rates = hyperparameters['merger_rate_density']
        return pp_rates[self.dataset_bins] + ln_dVTc

def get_posterior_resampling_weights(hyperposterior, nsamples, pixelpop_data, verbose=True):
    '''
    
    '''
    if verbose:
        print('='*50)
        print('Resampling GW posteriors')
        print('='*50 + '\n')
        
    ratefunc = PixelPopRateFunction(pixelpop_data, dataset_type='posteriors')
    @jax.jit
    def f(s):
        return ratefunc(pixelpop_data.posteriors, s)

    ss = [{k: hyperposterior[k][ii] for k in hyperposterior.keys()} for ii in range(nsamples)]
    rates = jnp.array(
        [jnp.array(f(ss[ii])) for ii in tqdm(range(nsamples))]
    )

    lweights = jnp.log(rates) - pixelpop_data.posteriors['log_prior'][None,...]

    normed_weights = lweights - LSE(lweights, axis=-1)[...,None] - jnp.log(lweights.shape[-1])
    normed_weights = jnp.exp(normed_weights)
    
    # compute neffs
    averaged_weights = jnp.mean(normed_weights, axis=0)
    design_effect = jnp.mean(averaged_weights**2, axis=-1)
    neffs = averaged_weights.shape[-1] / design_effect
    
    return normed_weights, neffs

def get_injection_resampling_weights(hyperposterior, nsamples, pixelpop_data, verbose=True):
    '''
    Docstring for get_injection_resampling_weights
    
    :param hyperposterior: Description
    :param nsamples: Description
    :param pixelpop_data: Description
    '''
    if verbose:
        print('='*50)
        print('Resampling detected injection set')
        print('='*50 + '\n')
        
    ratefunc = PixelPopRateFunction(pixelpop_data, dataset_type='injections')
    @jax.jit
    def f(s):
        return ratefunc(pixelpop_data.injections, s)

    ss = [{k: hyperposterior[k][ii] for k in hyperposterior.keys()} for ii in range(nsamples)]
    rates = jnp.array(
        [jnp.array(f(ss[ii])) for ii in tqdm(range(nsamples))]
    )

    lweights = jnp.log(rates) - pixelpop_data.injections['log_prior'][None,...]

    normed_weights = lweights - LSE(lweights, axis=-1)[...,None] - jnp.log(lweights.shape[-1])
    normed_weights = jnp.exp(normed_weights)
    
    # compute neffs
    averaged_weights = jnp.mean(normed_weights, axis=0)
    design_effect = jnp.mean(averaged_weights**2, axis=-1)
    neff = averaged_weights.shape[-1] / design_effect
    
    return normed_weights, neff

def reweight_events_and_injections(popsummary_result, hyperposterior, pixelpop_data, resampling_key=jax.random.PRNGKey(0), overwrite=False, verbose=True):
    '''
    Docstring for reweight_events_and_injections
    
    :param popsummary_result: Description
    :param hyperposterior: Description
    :param pixelpop_data: Description
    '''
        
    # flatten chains, if appropriate
    if isinstance(hyperposterior, list):
        processed_posterior = {}
        keys = hyperposterior[0].keys()
        for k in keys:
            stacked = jnp.stack([chain[k] for chain in hyperposterior])
            processed_posterior[k] = stacked
        hyperposterior = processed_posterior
    elif isinstance(hyperposterior, dict):    
        pass
    else:
        raise IOError(f'hyperposterior must be a list of dictionaries or single dictionary, you input {type(hyperposterior)}')
    hyperposterior, nsamples = pixelpop_data.fill_out_hyperposterior(hyperposterior)

    event_prng, inj_prng = jax.random.split(resampling_key)
    
    # reweight gw events, one sample per event to avoid bloat 
    event_w, event_neffs = get_posterior_resampling_weights(hyperposterior, nsamples, pixelpop_data, verbose=verbose)
    key = event_prng
    
    if verbose:
        print(f"Minimum population reweighted effective sample size is {np.min(event_neffs)} for event {popsummary_result.get_metadata('events')[np.argmin(event_neffs)]}")
    
    reweighted_events = []
    gw_parameters = popsummary_result.get_metadata('event_parameters')
    for ii, event in popsummary_result.get_metadata('events'):
        neff = event_neffs[ii]
        if verbose and neff < 100:
            warnings.warn(f"{event} population reweighted effective sample size is {neff}")
        
        weights = event_w[:,ii,...]
        sel = []
        for ii in range(weights.shape[0]):
            _, key = jax.random.split(key)
            down_selected = jax.random.choice(key, weights.shape[1], p=weights[ii])
            sel.append(down_selected)

        sel = np.array(sel)
        reweighted_event = np.array([
            pixelpop_data.posteriors[gw_parameter][sel] for gw_parameter in gw_parameters
        ]).T
        reweighted_events.append(reweighted_event[None,...]) # 1 sample per hypersample to avoid bloat
    reweighted_events = np.array(reweighted_events)
    
    popsummary_result.set_reweighted_event_samples(
        reweighted_event_samples = reweighted_events,
        overwrite=overwrite
    )
    
    # reweight injection set
    n_events = len(popsummary_result.get_metadata('events'))
    inj_w, inj_neff = get_injection_resampling_weights(hyperposterior, nsamples, pixelpop_data, verbose=verbose)
    
    key = inj_prng
    if verbose:
        print(f"injection set population reweighted effective sample size is {inj_neff}")
    
    sel = []
    for ii in range(inj_w.shape[0]):
        _, key = jax.random.split(key)
        down_selected = jax.random.choice(key, weights.shape[1], (n_events,), p=weights[ii])
        sel.append(down_selected)

    sel = np.array(sel)
    reweighted_injections = np.array([
        pixelpop_data.injections[gw_parameter][sel] for gw_parameter in gw_parameters
    ]).swapaxes(0,-1)
    
    popsummary_result.set_reweighted_injections(
        reweighted_injections = reweighted_injections[:,None,...], # just one catalog per hypersample
        overwrite=overwrite
    )
    
    