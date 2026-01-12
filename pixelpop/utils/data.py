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
            mean = 0.5*(minimum + maximum) # arithmetic mean    
            data[par] = jnp.where(bad, mean*jnp.ones_like(m), data[par])
            data['log_prior'] = jnp.where(bad, jnp.inf, data['log_prior'])
    return data

def check_bins(event_bins, injection_bins, bins=100):
    """
    Validate consistency between posterior-sample bins and injection bins.

    This function checks whether any posterior samples fall into bins that
    contain no injections, which would render Monte Carlo likelihood estimates
    unstable (formally divergent). It also verifies that both posterior and
    injection samples lie within the allowed bin range.

    Samples that violate these conditions are flagged by assigning an infinite
    prior weight, ensuring they do not contribute to Monte Carlo integrals.

    Parameters
    ----------
    event_bins : tuple of jax.numpy.ndarray
        Tuple of integer-valued bin indices for posterior samples, one array
        per dimension.
    injection_bins : tuple of jax.numpy.ndarray
        Tuple of integer-valued bin indices for injection samples, one array
        per dimension.
    bins : int or tuple of int, optional
        Number of bins per dimension. If an integer is provided, the same
        number of bins is assumed for all dimensions. Default is 100.

    Returns
    -------
    success : bool
        True if all checks pass; False if any invalid or injection-free bins
        are detected.
    problematic_posterior_samples : jax.numpy.ndarray
        Array marking posterior samples that fall outside the allowed range
        or into bins with no injections, set to `jnp.inf` where problematic.
    problematic_injections : jax.numpy.ndarray
        Array marking injection samples that fall outside the allowed range,
        set to `jnp.inf` where problematic.
    """

    if (not isinstance(event_bins, tuple)) or (not isinstance(injection_bins, tuple)) :
        warnings.warn('Bin check not implemented for flattened PixelPop')
        return True

    if isinstance(bins, int):
        bins = (bins,)*len(event_bins)

    problematic_posterior_samples = jnp.zeros_like(event_bins[0], dtype='float32')
    problematic_injections = jnp.zeros_like(injection_bins[0], dtype='float32')
    
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

    print(problematic_posterior_samples)
    return success, problematic_posterior_samples, problematic_injections
            

def place_in_bins(parameters, posteriors, injections, bins=100, minima={}, maxima={}, exit_on_error=False):
    """
    Discretize posterior and injection samples onto a common multidimensional bin grid.

    This function constructs a rectangular binning over the specified population
    parameters, places both posterior samples and injection samples into these
    bins, and performs consistency checks to ensure that all posterior bins are
    populated by injections. Bin ranges are taken from the default BBH population
    limits and can be overridden by user-supplied minima and maxima.

    Invalid samples or samples falling into injection-free bins are flagged via
    infinite prior weights to prevent numerical instabilities in Monte Carlo
    likelihood evaluations.

    Parameters
    ----------
    parameters : sequence of str
        Names of population parameters to bin. The order defines the bin axes.
    posteriors : dict-like
        Mapping from parameter names to posterior sample arrays.
    injections : dict-like
        Mapping from parameter names to injection sample arrays.
    bins : int or sequence of int, optional
        Number of bins per parameter. If a single integer is provided, the same
        number of bins is used for all dimensions. Default is 100.
    minima : dict, optional
        Dictionary of parameter-specific lower bounds overriding the defaults.
    maxima : dict, optional
        Dictionary of parameter-specific upper bounds overriding the defaults.
    exit_on_error : bool, optional
        If True, raise an exception when incompatible bins are detected.
        Otherwise, issue a warning and mask problematic samples. Default is False.

    Returns
    -------
    event_bins : tuple of jax.numpy.ndarray
        Bin indices for posterior samples, one array per parameter.
    inj_bins : tuple of jax.numpy.ndarray
        Bin indices for injection samples, one array per parameter.
    bin_axes : list of jax.numpy.ndarray
        Bin edge arrays for each parameter.
    logdV : jax.numpy.ndarray
        Logarithm of the bin volumes for each dimension.
    e_prior_mod : jax.numpy.ndarray
        Prior modifier for posterior samples, with `jnp.inf` marking invalid
        or injection-free bins.
    i_prior_mod : jax.numpy.ndarray
        Prior modifier for injection samples, with `jnp.inf` marking samples
        outside the allowed bin ranges.
    """

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