import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as LSE
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
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
    ratefunc = PixelPopRateFunction(pixelpop_data, dataset_type='posteriors')
    @jax.jit
    def f(s):
        return ratefunc(pixelpop_data.posteriors, s)

    ss = [{k: hyperposterior[k][ii] for k in hyperposterior.keys()} for ii in range(nsamples)]
    averaged_weights = np.zeros_like(pixelpop_data.posteriors['log_prior'])
    reweight_iloc = []
    loop = range(nsamples)
    if verbose:
        loop = tqdm(loop)
        loop.set_description('Resampling GW posteriors')
    for ii in loop:
        rates = np.array(f(ss[ii]))
        with np.errstate(divide='ignore'): # ignore np.log(0) 
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

    ratefunc = PixelPopRateFunction(pixelpop_data, dataset_type='injections')
    @jax.jit
    def f(s):
        return ratefunc(pixelpop_data.injections, s)

    ss = [{k: hyperposterior[k][ii] for k in hyperposterior.keys()} for ii in range(nsamples)]

    averaged_weights = np.zeros_like(pixelpop_data.injections['log_prior'])
    reweight_iloc = []

    loop = range(nsamples)
    if verbose:
        loop = tqdm(loop)
        loop.set_description('Resampling GW detected injections')

    for ii in loop:
        rates = np.array(f(ss[ii]))
        with np.errstate(divide='ignore'): # ignore np.log(0) 
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
        if p in map_to_gwpop_parameters:
            gw_parameters += map_to_gwpop_parameters[p]
        else:
            gw_parameters += [p]
    
    if len(gw_parameters) != len(popsummary_result.get_metadata('event_parameters')):        
        print(f'Updating event parameters from\n{popsummary_result.get_metadata("event_parameters")}\nto\n{gw_parameters}\n')
        popsummary_result.set_metadata('event_parameters', gw_parameters, overwrite=overwrite)

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
    
def sample_nd_grid(*bins, p, size=1):
    """
    Sample points from a piecewise constant probability density on an N-D grid.

    This function performs inverse transform sampling over a discrete grid 
    defined by `p`, then applies a uniform 'intra-bin scatter' to provide 
    continuous samples within each cell. 

    Parameters
    ----------
    *bins : array_like
        1-D arrays representing the bin edges or bin centers for each 
        dimension. The number of positional arguments defines the 
        dimensionality of the space.
    p : ndarray
        An N-dimensional array of weights (densities) corresponding to the 
        grid defined by `bins`. The shape of `p` must match the lengths of 
        the input bins.
    size : int, optional
        The number of samples to generate. Default is 1.

    Returns
    ----------
    samples : generator of ndarrays
        A generator yielding one array for each dimension. Each array has 
        length `size`, representing the coordinates of the sampled points.

    Notes
    ----------
    The function assumes the bins are equally spaced for the calculation 
    of `dx`. The total probability is normalized internally by 
    dividing `p` by its sum.

    Examples
    ----------
    >>> x_bins = np.linspace(0, 1, 10)
    >>> y_bins = np.linspace(0, 1, 10)
    >>> weights = np.random.rand(10, 10)
    >>> x_samples, y_samples = sample_nd_grid(x_bins, y_bins, p=weights, size=1000)
    """
    dimension = len(bins)
    dx = [bins[ii][1] - bins[ii][0] for ii in range(dimension)]
    shape = tuple(len(b) for b in bins)

    pflatten = p.reshape(-1)/np.sum(p)
    which = np.random.choice(len(pflatten), size=size, replace=True, p=pflatten)
    which = np.unravel_index(which, shape=shape)

    intra_bin_scatter = [d*np.random.uniform(size=size) for d in dx]

    samples = tuple(b[which[ii]] + intra_bin_scatter[ii] for ii, b in enumerate(bins))
    return samples

def Spearman_Sample(log_pxy, x_bins, y_bins, precision=int(4e4), x_range=(0, 1), y_range=(0, 1)):
    """
    Compute Spearman correlations from a 2D log-probability grid.

    Parameters
    ----------
    log_pxy : ndarray
        2D array of log-probabilities (densities).
    x_bins, y_bins : ndarray
        1-D arrays representing the grid edges/bins for each dimension.
    precision : int, optional
        Number of samples to draw for the correlation estimate. Default is 40,000.
    x_range, y_range : tuple of float, optional
        Fractional window [0, 1] to sub-select the grid before computing correlation.

    Returns
    -------
    rho : float
        Spearman's rank correlation coefficient between x and y.
    rho_var : float
        Spearman's rank correlation between x and the variance of y, 
        often used as a diagnostic for heteroscedasticity.
    """
    if log_pxy.ndim != 2:
        raise ValueError(f"Expected 2D array for log_pxy, got {log_pxy.ndim}D.")

    # define windowed space
    nx, ny = log_pxy.shape
    ix_min, ix_max = int(x_range[0] * nx), int(x_range[1] * nx)
    iy_min, iy_max = int(y_range[0] * ny), int(y_range[1] * ny)

    log_slice = log_pxy[ix_min:ix_max, iy_min:iy_max]
    pxy_slice = np.exp(log_slice - np.max(log_slice)) # subtract max for stability
    
    sub_x_bins = x_bins[ix_min:ix_max]
    sub_y_bins = y_bins[iy_min:iy_max]

    x_samples, y_samples = sample_nd_grid(sub_x_bins, sub_y_bins, p=pxy_slice, size=precision)
        
    rho, _ = spearmanr(x_samples, y_samples)
    
    y_sq_dev = (y_samples - np.mean(y_samples))**2
    rho_var, _ = spearmanr(x_samples, y_sq_dev)

    return rho, rho_var