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

    problematic_posterior_samples = jnp.zeros_like(event_bins[0])
    problematic_injections = jnp.zeros_like(injection_bins[0])
    print(problematic_posterior_samples.shape, problematic_injections.shape)
    # first check if any -1 or (bins=100) in the list
    success = True
    for ii, b in enumerate(event_bins):
        bad = jnp.logical_or(b == -1, b == bins[ii])
        if jnp.any(bad):
            warnings.warn('Some posterior samples are outside the PixelPop range. User should clean samples.')
            success = False
            problematic_posterior_samples = problematic_posterior_samples[bad].set(jnp.inf)
    for ii, b in enumerate(injection_bins):
        bad = jnp.logical_or(b == -1, b == bins[ii])
        if jnp.any(bad):
            warnings.warn('Some injection samples are outside the PixelPop range. User should clean samples.')
            success = False
            problematic_injections = problematic_injections[bad].set(jnp.inf)
    
    # check if any posterior samples are in injection-free bins. Causes instabilities in PixelPop
    
    # first uniquely flatten bins
    # flatten to single index for each bin to assist checking of uniqueness. Simpler than a multi-dimensional index
    flattened_ebins = jnp.ravel_multi_index(event_bins, bins)
    flattened_ibins = jnp.ravel_multi_index(injection_bins, bins)

    isin = jnp.isin(flattened_ebins, flattened_ibins)
    if jnp.any(~isin):
        warnings.warn(
            f'\n\tSome ({jnp.sum(~isin)}, {int(10_000*jnp.mean(~isin)+0.001)/100}%) posterior samples are in bins with no detectability.\n',
            RuntimeWarning,
            stacklevel=1
            )
        worst_ev_i, worst_ev = jnp.argsort(jnp.mean(~isin, axis=1))[-3:], jnp.array(jnp.sort(1e4*jnp.mean(~isin, axis=1))[-3:], dtype=int)/100
        warnings.warn(
            f'\n\tEvent #{worst_ev_i} has {worst_ev}% posterior samples in bins with no detectability.\n',
            RuntimeWarning,
            stacklevel=1
            )
        success = False
        problematic_posterior_samples = problematic_posterior_samples.at[~isin].set(jnp.inf)

    return success, problematic_posterior_samples, problematic_injections
            

def place_in_bins(parameters, posteriors, injections, bins=100, minima={}, maxima={}, exit_on_error=False):

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

    success, e_prior_mod, i_prior_mod = check_bins(event_bins, inj_bins, bins)
    if not success:
        if exit_on_error:
            raise IndexError('Some event indices incompatible with injection indices in PixelPop.')
        else:
            warnings.warn(
                '\n\tSome event indices incompatible with injection indices in PixelPop, setting prior values to jnp.inf\n',
                RuntimeWarning,
                stacklevel=6
                )

    return event_bins, inj_bins, bin_axes, logdV, e_prior_mod, i_prior_mod