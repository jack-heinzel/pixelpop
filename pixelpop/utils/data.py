from jax import numpy as jnp
from . import place_samples_in_bins
from ..models import gwpop_models
import warnings

def convert_m1q_to_lm1m2(data):
    m1 = data.pop('mass_1')
    q = data.pop('mass_ratio')

    data['log_mass_1'] = jnp.log(m1)
    data['log_mass_2'] = data['log_mass_1'] + jnp.log(q)
    data['log_prior'] = jnp.log(data.pop('prior')) + data['log_mass_2']
    return data

def convert_m1_to_lm1(data):
    m1 = data.pop('mass_1')
    data['log_mass_1'] = jnp.log(m1)
    data['log_prior'] = jnp.log(data.pop('prior')) + data['log_mass_1']
    return data

def convert_m1m2_to_lm1lm2(data):
    m1 = data.pop('mass_1')
    data['log_mass_1'] = jnp.log(m1)
    m2 = data.pop('mass_2')
    data['log_mass_2'] = jnp.log(m2)
    data['log_prior'] = jnp.log(data.pop('prior')) + data['log_mass_1'] + data['log_mass_2']
    return data

def clean_par(data, par, minimum, maximum, remove=False):
    if par in data:
        m = data[par]
        bad = jnp.logical_or(m < minimum, m > maximum)
        if remove:
            for k in data:
                try:
                    data[k] = data[k][~bad]
                except (TypeError, IndexError):
                    continue
        else:
            geom = 0.5*(minimum + maximum) # geometric mean    
            data[par] = jnp.where(bad, geom*jnp.ones_like(m), data[par])
            data['log_prior'] = jnp.where(bad, jnp.inf, data['log_prior'])
    return data

def check_bins(event_bins, injection_bins, bins=100):
    if (not isinstance(event_bins, tuple)) or (not isinstance(injection_bins, tuple)) :
        warnings.warn('Bin check not implemented for flattened PixelPop')
        return True

    if isinstance(bins, int):
        bins = (bins,)*len(event_bins)

    # first check if any -1 or (bins=100) in the list
    success = True
    for ii, b in enumerate(event_bins):
        if jnp.any(jnp.logical_or(b == -1, b == bins[ii])):
            warnings.warn('Some posterior samples are outside the PixelPop range. User should clean samples.')
            success = False
    for ii, b in enumerate(injection_bins):
        if jnp.any(jnp.logical_or(b == -1, b == bins[ii])):
            warnings.warn('Some injection samples are outside the PixelPop range. User should clean samples.')
            success = False
    
    # check if any posterior samples are in injection-free bins. Causes instabilities in PixelPop
    
    # first uniquely flatten bins
    # flatten to single index for each bin to assist checking of uniqueness. Simpler than a multi-dimensional index
    flattened_ebins = jnp.ravel_multi_index(event_bins, bins)
    flattened_ibins = jnp.ravel_multi_index(injection_bins, bins)

    isin = jnp.isin(flattened_ebins, flattened_ibins, kind='table')
    if jnp.any(~isin):
        warnings.warn(
            'Some posterior samples are in bins with no detectability.' \
            'PixelPop is unstable to these Monte Carlo issues, and may put a spike at this bin.'
            )
        success = False

    return success
            

def place_in_bins(parameters, posteriors, injections, bins=100, minima={}, maxima={}, exit_on_error=True):

    bbh_minima = gwpop_models.bbh_minima.copy()
    bbh_maxima = gwpop_models.bbh_maxima.copy()
    
    bbh_minima.update(minima)
    bbh_maxima.update(maxima)
    if jnp.ndim(bins) == 0:
        bins = [bins] * len(parameters)

    bin_axes = [jnp.linspace(bbh_minima[par], bbh_maxima[par], bins[ii]+1) for ii, par in enumerate(parameters)]
    logdV = jnp.log(jnp.array([b[1] - b[0] for b in bin_axes]))

    sample_coordinates = [posteriors[par] for par in parameters]
    event_bins = place_samples_in_bins(bin_axes, sample_coordinates) 

    # places VT injection set in bins
    inj_coordinates = [injections[par] for par in parameters]
    inj_bins = place_samples_in_bins(bin_axes, inj_coordinates)

    # TODO: implement check for bin = -1 or bin = maximum (100=bins)
    # TODO: implement check if event bins where inj bins are not
    success = check_bins(event_bins, inj_bins, bins)
    if exit_on_error and (not success):
        raise IndexError('Event indices incompatible with injection indices in PixelPop.')

    return event_bins, inj_bins, bin_axes, logdV