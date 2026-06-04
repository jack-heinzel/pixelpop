from jax import numpy as jnp
import numpy as np
from . import place_samples_in_bins
from ..models import gwpop_models
import warnings

from .nearest_neighbor import create_CAR_coupling_matrix
from dataclasses import dataclass, field
from typing import Dict, List, Union, Callable, Any, Tuple, Optional

import numpyro.distributions as dist

def convert_m1q_to_lm1m2(data):
    m1 = data.pop('mass_1')
    q = data.pop('mass_ratio')

    data['log_mass_1'] = jnp.log(m1)
    data['log_mass_2'] = data['log_mass_1'] + jnp.log(q)
    data['log_prior'] = jnp.log(data.pop('prior')) + data['log_mass_2']
    return data

def convert_m1q_to_lm1lm2(data):
    # without typo in name
    return convert_m1q_to_lm1m2(data)

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
            problematic_posterior_samples = problematic_posterior_samples.at[bad].set(jnp.inf)
    for ii, b in enumerate(injection_bins):
        bad = jnp.logical_or(b == -1, b == bins[ii])
        if jnp.any(bad):
            warnings.warn('Some injection samples are outside the PixelPop range. User should clean samples.')
            success = False
            problematic_injections = problematic_injections.at[bad].set(jnp.inf)
    
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

    
    if np.ndim(bins) == 0:
        bins = [bins] * len(parameters)

    bin_axes = [jnp.linspace(minima[par], maxima[par], bins[ii]+1) for ii, par in enumerate(parameters)]
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


def build_bin_axes(parameters, bins, minima, maxima):
    """
    Construct the bin edges and log bin volumes for the pixelized grid.

    This depends only on the grid definition (parameters, number of bins, and
    axis limits), not on any particular dataset, so it is computed once and reused
    for both injections and every event.

    Parameters
    ----------
    parameters : sequence of str
        Names of the binned parameters. The order defines the bin axes.
    bins : int or sequence of int
        Number of bins per parameter. A scalar is broadcast to all dimensions.
    minima, maxima : dict
        Parameter -> lower/upper bound of the grid.

    Returns
    -------
    bin_axes : list of jax.numpy.ndarray
        Bin edge arrays, one per parameter.
    logdV : jax.numpy.ndarray
        Logarithm of the bin width along each dimension.
    """
    if np.ndim(bins) == 0:
        bins = [bins] * len(parameters)
    bin_axes = [jnp.linspace(minima[par], maxima[par], bins[ii] + 1) for ii, par in enumerate(parameters)]
    logdV = jnp.log(jnp.array([b[1] - b[0] for b in bin_axes]))
    return bin_axes, logdV


def bin_dataset(parameters, dataset, bin_axes):
    """
    Place a single dataset's samples into the pixelized grid.

    Parameters
    ----------
    parameters : sequence of str
        Names of the binned parameters, matching ``bin_axes``.
    dataset : dict-like
        Mapping from parameter name to a 1-D array of samples.
    bin_axes : list of jax.numpy.ndarray
        Bin edge arrays, e.g. from :func:`build_bin_axes`.

    Returns
    -------
    bins : tuple of jax.numpy.ndarray
        Bin indices, one array per parameter.
    """
    coordinates = [dataset[par] for par in parameters]
    return place_samples_in_bins(bin_axes, coordinates)


def flag_out_of_range(sample_bins, bins):
    """
    Build an additive log-prior modifier flagging samples outside the grid.

    A sample is out of range when any of its bin indices is ``-1`` (below the grid)
    or equal to the number of bins along that axis (above the grid). Such samples
    are assigned an infinite log-prior so they drop out of Monte Carlo sums.

    Parameters
    ----------
    sample_bins : tuple of jax.numpy.ndarray
        Bin indices for one dataset, one array per parameter.
    bins : int or sequence of int
        Number of bins per parameter.

    Returns
    -------
    prior_mod : jax.numpy.ndarray
        Array shaped like a single bin index array, ``jnp.inf`` at out-of-range
        samples and ``0`` elsewhere.
    """
    if np.ndim(bins) == 0:
        bins = [bins] * len(sample_bins)
    prior_mod = jnp.zeros_like(sample_bins[0], dtype='float32')
    for ii, b in enumerate(sample_bins):
        bad = jnp.logical_or(b == -1, b == bins[ii])
        prior_mod = jnp.where(bad, jnp.inf, prior_mod)
    return prior_mod


def flag_injection_free(event_bins, inj_bins, bins):
    """
    Build an additive log-prior modifier flagging event samples in empty bins.

    Posterior samples that land in bins containing no injections make the Monte
    Carlo selection estimate formally divergent, so they are assigned an infinite
    log-prior. Works on a single event's (1-D) bin indices.

    Parameters
    ----------
    event_bins : tuple of jax.numpy.ndarray
        Bin indices for one event, one array per parameter.
    inj_bins : tuple of jax.numpy.ndarray
        Bin indices for the injection set, one array per parameter.
    bins : int or sequence of int
        Number of bins per parameter.

    Returns
    -------
    prior_mod : jax.numpy.ndarray
        ``jnp.inf`` at samples falling in injection-free bins, ``0`` elsewhere.
    """
    if np.ndim(bins) == 0:
        bins = [bins] * len(event_bins)
    bins = tuple(bins)
    # 'clip' keeps out-of-range indices (already flagged by flag_out_of_range)
    # from raising; their value is irrelevant since their prior is inf anyway.
    flat_e = jnp.ravel_multi_index(event_bins, bins, mode='clip')
    flat_i = jnp.ravel_multi_index(inj_bins, bins, mode='clip')
    isin = jnp.isin(flat_e, flat_i)
    return jnp.where(isin, 0.0, jnp.inf)


# Assuming you have your COSMO object available globally or pass it in
# from .cosmology import COSMO

@dataclass
class PixelPopData:
    """
    Helper class which holds data:
    - Single event posteriors
    - Injection set
    - PixelPop specific arguments:
        - PixelPop parameters
        - Other "nuisance" parameters
        - bins (number along each axis)
        - axis minima and maxima
    - Analysis settings
        - variance cut
        - lower triangular flag (for m1, m2 analyses)
        - length_scales flag
        - marginalize_sigma flag
    - Additional settings or flags, which should usually be set to defaults:
        - random_initialization
        - plausible_hyperparameters
        - skip_nonparametric
        - constraint_functions
        - coupling_prior

    Parameters
    ----------
    name : str
        name for saving result files and chains
    posteriors : list of dict
        Per-event posterior samples as a pytree: a list with one dict per event,
        each dict mapping parameter name to a 1-D array of that event's samples.
        Different events may have different numbers of samples (ragged). Each event
        dict must also include 'log_prior'.
    injections : dict
        Injection data keyed by parameter name. Each entry is shaped (Nfound).
        Must include 'ln_prior', 'total_generated' (int/float), and
        'analysis_time' (float).
    pixelpop_parameters : list of str
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
    parameter_to_hyperparameters : dict, optional
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
    IID : bool, optional
        If True, the merger rate field is evaluated on both BHs 1 and 2
    cauchy_icar : bool, optional (EXPERIMENTAL)
        If True, use Cauchy ICAR coupling prior, more sensitivity to gaps and more robust uncertainties
    skip_nonparametric : bool, optional
        If True, disable the pixelized (nonparametric) component.
    constraint_funcs : list of callables, optional
        Extra constraint functions applied to hyperparameters.
    marginalize_sigma : bool, optional
        If True, PixelPop analysis uses an analytic marginalization over the sigma 
        coupling strength parameter. Can only be done if length_scales = False. 
        Typically, this improves chain convergence.

    """
    name: str
    # Data
    posteriors: List[Dict[str, Any]]
    injections: Dict[str, Any]
    
    # Gravitational wave parameter space
    pixelpop_parameters: List[str] 
    other_parameters: List[str]
    bins: Union[int, List[int]]
    
    # Axis limits
    minima: Dict[str, float] = field(default_factory=dict)
    maxima: Dict[str, float] = field(default_factory=dict)

    # Models and priors
    parametric_models: Dict[str, Callable] = field(default_factory=dict)
    parameter_to_hyperparameters: Dict[str, List[str]] = field(default_factory=dict)
    priors: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis settings
    UncertaintyCut: float = 1.0
    lower_triangular: bool = False
    cauchy_icar: bool = False
    marginalize_sigma: bool = False
    length_scales: bool = False
    IID: bool = False # TODO: make this IID parameters, so some parameters can be IID others not (e.g., a1, a2 IID, mass ratio not)
    EventNeffCut: float = 0.
    SelectionNeffCut: bool = False
    # Additional settings
    random_initialization: bool = True
    plausible_hyperparameters: Dict[str, float] = field(default_factory=dict)
    skip_nonparametric: bool = False
    constraint_funcs: List[Callable] = field(default_factory=list)
    coupling_prior: Tuple[Any, Any] = ((-3, 3), dist.Uniform)

    def preprocess_cosmology(self, cosmology):
        """
        Calculates differential comoving volumes if 'redshift' is a parameter.
        Modifies self.posteriors and self.injections in-place to add 'ln_dVTc'.
        """
        
        print("Preprocessing cosmology data...")
        # from unxt.quantity import Quantity

        # Extract data
        inj_z = self.injections['redshift']

        max_event_z = max(float(jnp.max(event['redshift'])) for event in self.posteriors)
        max_z = np.maximum(np.max(inj_z), max_event_z)
        zs = np.linspace(1e-6, max_z, 10000)
        
        # Calculate dVc/dz / (1+z)
        dVs = cosmology.differential_comoving_volume(zs)
        
        # if isinstance(dVs, Quantity):
        #     # TODO: implement in terms of unxt/wcosmo unit manipulations
        #     dVs = dVs.value 
        try:
            dVs = dVs.value
        except AttributeError:
            pass
        dVs = 4 * np.pi * 1e-9 * dVs 
            
        ln_dVTc = np.log(dVs) - np.log(1 + zs)

        # Interpolate and store, per event, in the dictionaries
        for event in self.posteriors:
            event['ln_dVTc'] = jnp.interp(event['redshift'], zs, ln_dVTc)
        self.injections['ln_dVTc'] = jnp.interp(inj_z, zs, ln_dVTc)
        
    def __post_init__(self):
        """
        Optional: Validation or automatic formatting after object creation.
        """
        if self.marginalize_sigma and self.length_scales:
            import warnings
            warnings.warn(
                "Using experimental grid-marginalized ICAR for per-dimension length scales. "
                "Grid bounds are taken from coupling_prior.",
                stacklevel=2,
            )
        self.Nobs = len(self.posteriors)
        # standardize bin dimension
        self.dimension = len(self.pixelpop_parameters)
        if jnp.ndim(self.bins) == 0:
            self.bins = [self.bins] * self.dimension

        # window function
        self.window_parameters = [p.replace('_window', '') for p in self.other_parameters if '_window' in p]
        self.has_window = len(self.window_parameters) > 0
        
        self.adj_matrices = [
            create_CAR_coupling_matrix(self.bins[ii], 1, isVisible=False) for ii in range(self.dimension)
            ]

        new_minima = gwpop_models.bbh_minima.copy()
        new_maxima = gwpop_models.bbh_maxima.copy()
        
        new_minima.update(self.minima)
        new_maxima.update(self.maxima)

        self.minima = new_minima
        self.maxima = new_maxima

        # bin up events and injections. Bin axes depend only on the grid, so they
        # are built once; injections are binned once; each (ragged) event is binned
        # in a loop, accumulating its own log-prior modifiers.
        any_problematic = False
        if self.IID:
            pp1 = [x + '_1' for x in self.pixelpop_parameters]
            pp2 = [x + '_2' for x in self.pixelpop_parameters]
            self.bin_axes, self.logdV = build_bin_axes(pp1, self.bins, self.minima, self.maxima)

            self.inj_bins_1 = bin_dataset(pp1, self.injections, self.bin_axes)
            self.inj_bins_2 = bin_dataset(pp2, self.injections, self.bin_axes)
            iprior = flag_out_of_range(self.inj_bins_1, self.bins) + flag_out_of_range(self.inj_bins_2, self.bins)
            self.injections['log_prior'] += iprior
            any_problematic |= bool(jnp.any(jnp.isinf(iprior)))

            self.event_bins_1, self.event_bins_2 = [], []
            for event in self.posteriors:
                eb1 = bin_dataset(pp1, event, self.bin_axes)
                eb2 = bin_dataset(pp2, event, self.bin_axes)
                eprior = (
                    flag_out_of_range(eb1, self.bins)
                    + flag_injection_free(eb1, self.inj_bins_1, self.bins)
                    + flag_out_of_range(eb2, self.bins)
                    + flag_injection_free(eb2, self.inj_bins_2, self.bins)
                )
                event['log_prior'] += eprior
                self.event_bins_1.append(eb1)
                self.event_bins_2.append(eb2)
                any_problematic |= bool(jnp.any(jnp.isinf(eprior)))

        else:
            self.bin_axes, self.logdV = build_bin_axes(
                self.pixelpop_parameters, self.bins, self.minima, self.maxima
            )

            self.inj_bins = bin_dataset(self.pixelpop_parameters, self.injections, self.bin_axes)
            iprior = flag_out_of_range(self.inj_bins, self.bins)
            self.injections['log_prior'] += iprior
            any_problematic |= bool(jnp.any(jnp.isinf(iprior)))

            self.event_bins = []
            for event in self.posteriors:
                eb = bin_dataset(self.pixelpop_parameters, event, self.bin_axes)
                eprior = flag_out_of_range(eb, self.bins) + flag_injection_free(eb, self.inj_bins, self.bins)
                event['log_prior'] += eprior
                self.event_bins.append(eb)
                any_problematic |= bool(jnp.any(jnp.isinf(eprior)))

        if any_problematic:
            warnings.warn(
                '\n\tSome samples fall outside the PixelPop range or in injection-free '
                'bins; their prior was set to jnp.inf so they drop out of Monte Carlo sums.\n',
                RuntimeWarning,
                stacklevel=2,
            )

        full_hyperparams = gwpop_models.gwparameter_to_hyperparameters.copy()
        full_hyperparams.update(self.parameter_to_hyperparameters)
        self.parameter_to_hyperparameters = full_hyperparams

        final_models = {}
        for p in self.other_parameters:
            if p in self.parametric_models:
                # User provided an override in the input dict
                print(f'Updating {p} model to {self.parametric_models[p].__name__}')
                final_models[p] = self.parametric_models[p]
            else:
                # Fall back to global default
                print(f'Using default {p} model {gwpop_models.gwparameter_to_model[p].__name__}')
                final_models[p] = gwpop_models.gwparameter_to_model[p]
        self.parametric_models = final_models

        final_priors = {}
        for p in self.other_parameters:
            
            for h in self.parameter_to_hyperparameters[p]:
                if h in self.priors:    
                    # User provided override
                    pprint = self.priors[h]
                    print(f'Using custom prior {h} = {pprint[1].__name__}{tuple(pprint[0])}')
                    final_priors[h] = self.priors[h]
                else:
                    # Global default
                    pprint = gwpop_models.default_priors[h]
                    print(f'Using default prior {h} = {pprint[1].__name__}{tuple(pprint[0])}')
                    final_priors[h] = gwpop_models.default_priors[h]
        self.priors = final_priors

        for p in self.window_parameters:
            if p + '_window' not in self.parametric_models:
                raise ValueError(f'Window parameter {p} not found in parametric_models')
            if p not in self.pixelpop_parameters:
                raise ValueError(f'Window parameter {p} not found in pixelpop_parameters')
        # for now, hardcode Planck15_LAL cosmology
        # TODO: allow for different cosmologies
        self.preprocess_cosmology(gwpop_models.COSMO)

    def fill_out_hyperposterior(self, hyperposterior):
        '''
        Helper function for adding delta parameters to the hyperposterior
            
        Parameters
        ----------
        hyperposterior : dict
            dictionary of hyperposterior samples, chains flattened
                
        Returns
        -------
        hyperposterior : dict
            hyperposterior with added samples at delta function prior sites
        '''
        delta_pars = {}
        for p in self.other_parameters:
            for h in self.parameter_to_hyperparameters[p]:
                if self.priors[h][1].__name__ == 'Delta':
                    delta_pars[h] = self.priors[h][0][0]
        key0 = list(hyperposterior.keys())[0]
        Nsamples = len(hyperposterior[key0])
        for par in self.other_parameters:
            required_keys = self.parameter_to_hyperparameters[par]
            for k in required_keys:
                if not k in hyperposterior:
                    hyperposterior[k] = delta_pars[k]*jnp.ones(Nsamples)
        return hyperposterior, Nsamples
