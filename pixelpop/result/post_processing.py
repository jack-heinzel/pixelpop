import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as LSE
from tqdm import tqdm
import numpy as np
import warnings
from functools import reduce
import os
from glob import glob
import h5ify
import pickle as pkl
import json
from ..models.gwpop_models import map_to_gwpop_parameters

def combine_chains(chain_1, chain_2):
    for k in chain_1.keys():
        assert k in chain_2.keys()
    assert len(chain_1.keys()) == len(chain_2.keys())

    x = {}
    for k in chain_1.keys():
        x[k] = np.concatenate((chain_1[k], chain_2[k]), axis=0)
    return x

def get_posterior(rundir, chain_regex='chain_*_samples', result_file_type='h5'):
    fpath = os.path.join(rundir, chain_regex+'.'+result_file_type)
    paths = glob(fpath)
    print(f"I got {len(paths)} unique chains")
    chains = []
    for p in paths:
        if result_file_type == 'h5':
            chain = h5ify.load(p)
        elif result_file_type == 'pkl':
            with open(p, 'rb') as ff:
                chain = pkl.load(ff)
        else:
            raise TypeError(
                'h5 and pkl are only accepted result_file_type'
                )
        chains.append(chain)

    return chains

def get_input_metadata(file_label, datadir='../data'):
    file_path = os.path.join(datadir, file_label, 'event_data.json')
    try:
        with open(file_path, 'r') as file:
            metadata = json.load(file)
    except FileNotFoundError:
        file_path = os.path.join(datadir, file_label, 'data', 'event_data.json')
        with open(file_path, 'r') as file:
            metadata = json.load(file)
    wf_paths = metadata.keys()
    wfs = []
    for p in wf_paths:
        if 'S230529' in p or 'GW230529' in p:
            wfs.append('GW230529')
            continue
        name = p.split('/')[-1]
        if name.startswith('S'):
            wfs.append(name.split('-')[0])
        else:
            wfs.append(name.split('-')[3].replace('_PEDataRelease_mixed_cosmo.h5', '').replace('_PEDataRelease_cosmo.h5', ''))
    # print(wfs)
    return wfs, wf_paths, metadata

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

def resample_posteriors(hyperposterior, nsamples, pixelpop_data, samples_per_event=1, verbose=True):
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
    averaged_weights = np.zeros_like(pixelpop_data.posteriors['log_prior'])
    reweight_iloc = []
    for ii in tqdm(range(nsamples)):
        rates = np.array(f(ss[ii]))
        lweights = np.log(rates) - np.array(pixelpop_data.posteriors['log_prior'])
        lnorms = LSE(lweights, axis=-1)
        normed_weights = lweights - lnorms[...,None]
        normed_weights = np.exp(normed_weights)
        averaged_weights += normed_weights
        
        sel = []
        for jj in range(normed_weights.shape[0]):
            down_selected = np.random.choice(normed_weights.shape[1], p=normed_weights[jj])
            sel.append(down_selected)

        reweight_iloc.append(sel)
    reweight_iloc = np.array(reweight_iloc)
    
    averaged_weights = averaged_weights / np.mean(averaged_weights, axis=-1)[...,None]
    design_effect = np.mean(averaged_weights**2, axis=-1)
    neffs = averaged_weights.shape[-1] / design_effect
    
    return reweight_iloc, neffs

def resample_injections(hyperposterior, nsamples, nevents, pixelpop_data, verbose=True):
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

    averaged_weights = np.zeros_like(pixelpop_data.injections['log_prior'])
    reweight_iloc = []
    for ii in tqdm(range(nsamples)):
        rates = np.array(f(ss[ii]))
        lweights = np.log(rates) - np.array(pixelpop_data.injections['log_prior'])
        lnorms = LSE(lweights, axis=-1)
        normed_weights = lweights - lnorms[...,None]
        normed_weights = np.exp(normed_weights)
        averaged_weights += normed_weights
        
        sel = np.random.choice(normed_weights.shape[0], size=nevents, p=normed_weights)

        reweight_iloc.append(sel)
    reweight_iloc = np.array(reweight_iloc)
    
    averaged_weights = averaged_weights / np.mean(averaged_weights, axis=-1)[...,None]
    design_effect = np.mean(averaged_weights**2, axis=-1)
    neff = averaged_weights.size / design_effect

    return reweight_iloc, neff

def reweight_events_and_injections(popsummary_result, hyperposterior, pixelpop_data, resampling_key=0, overwrite=False, verbose=True):
    '''
    Docstring for reweight_events_and_injections
    
    :param popsummary_result: Description
    :param hyperposterior: Description
    :param pixelpop_data: Description
    '''
        
    # flatten chains, if appropriate
    if isinstance(hyperposterior, list):
        hyperposterior = reduce(combine_chains, hyperposterior)
    elif isinstance(hyperposterior, dict):    
        pass
    else:
        raise IOError(f'hyperposterior must be a list of dictionaries or single dictionary, you input {type(hyperposterior)}')
    hyperposterior, nsamples = pixelpop_data.fill_out_hyperposterior(hyperposterior)

    np.random.seed(resampling_key)
    
    # reweight gw events, one sample per event to avoid bloat 
    event_iloc, event_neffs = resample_posteriors(hyperposterior, nsamples, pixelpop_data, verbose=verbose)
    if verbose:
        print(f"Minimum population reweighted effective sample size is {np.min(event_neffs)} for event {popsummary_result.get_metadata('events')[np.argmin(event_neffs)]}")
    
    reweighted_events = []
    gw_parameters = []
    for p in popsummary_result.get_metadata('event_parameters'):
        gw_parameters += map_to_gwpop_parameters[p]
    
    if len(gw_parameters) != len(popsummary_result.get_metadata('event_parameters')):        
        print(f'Updating event parameters from\n{popsummary_result.get_metadata("event_parameters")}\nto\n{gw_parameters}\n')
        popsummary_result.set_metadata('event_parameters', gw_parameters, overwrite=overwrite)
    print('here')
    for ii, event in enumerate(popsummary_result.get_metadata('events')):
        neff = event_neffs[ii]
        if verbose and neff < 100:
            warnings.warn(f"{event} population reweighted effective sample size is {neff}")

        reweighted_event = np.array([
            pixelpop_data.posteriors[gw_parameter][ii,event_iloc[:,ii]] for gw_parameter in gw_parameters
        ]).T        
        reweighted_events.append(reweighted_event[None,...]) # 1 sample per hypersample to avoid bloat
    
    reweighted_events = np.array(reweighted_events) 
    popsummary_result.set_reweighted_event_samples(
        reweighted_event_samples = reweighted_events,
        overwrite = overwrite
    )
    print('done')
    
    # reweight injection set
    n_events = len(popsummary_result.get_metadata('events'))
    inj_iloc, inj_neff = resample_injections(hyperposterior, nsamples, n_events, pixelpop_data, verbose=verbose)
    
    if verbose:
        print(f"injection set population reweighted effective sample size is {inj_neff}")
    
    reweighted_injections = np.array([
        pixelpop_data.injections[gw_parameter][inj_iloc] for gw_parameter in gw_parameters
    ]).swapaxes(0,-1)
    
    popsummary_result.set_reweighted_injections(
        reweighted_injections = reweighted_injections[:,None,...], # just one catalog per hypersample
        overwrite=overwrite
    )
    
    