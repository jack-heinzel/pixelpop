from jax import numpy as jnp
from . import place_samples_in_bins

def convert_m1q_to_lm1m2(data, min_m=3, max_m=200):
    m1 = data.pop('mass_1')
    q = data.pop('mass_ratio')

    data['log_mass_1'] = jnp.log(m1)
    data['log_mass_2'] = data['log_mass_1'] + jnp.log(q)
    data['log_prior'] = jnp.log(data.pop('prior')) + data['log_mass_2']

def convert_m1_to_lm1(data):
    m1 = data.pop('mass_1')
    data['log_mass_1'] = jnp.log(m1)
    data['log_prior'] = jnp.log(data.pop('prior')) + data['log_mass_1']

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

def place_in_bins(parameters, posteriors, injections, bins=100, minima={}, maxima={}):

    bbh_minima = {'log_mass_1': jnp.log(3), 'mass_ratio': 0., 'log_mass_2': jnp.log(3), 'chi_eff': -1., 'redshift': 0.}
    bbh_maxima = {'log_mass_1': jnp.log(200), 'mass_ratio': 1., 'log_mass_2': jnp.log(200), 'chi_eff': 1., 'redshift': 2.4}
    
    bbh_minima.update(minima)
    bbh_maxima.update(maxima)

    bin_axes = [jnp.linspace(bbh_minima[par], bbh_maxima[par], bins+1) for par in parameters]
    logdV = jnp.log(jnp.array([b[1] - b[0] for b in bin_axes]))

    sample_coordinates = [posteriors[par] for par in parameters]
    event_bins = place_samples_in_bins(bin_axes, sample_coordinates) 

    # places VT injection set in bins
    inj_coordinates = [injections[par] for par in parameters]
    inj_bins = place_samples_in_bins(bin_axes, inj_coordinates)

    return event_bins, inj_bins, bin_axes, logdV