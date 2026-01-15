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

    
    if jnp.ndim(bins) == 0:
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
    posteriors : dict
        Posterior samples keyed by parameter name. Each entry is shaped
        (Nobs, Nsample). Must also include 'ln_prior'.
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
    skip_nonparametric : bool, optional
        If True, disable the pixelized (nonparametric) component.
    constraint_funcs : list of callables, optional
        Extra constraint functions applied to hyperparameters.
    marginalize_sigma : bool, optional
        If True, PixelPop analysis uses an analytic marginalization over the sigma 
        coupling strength parameter. Can only be done if length_scales = False. 
        Typically, this improves chain convergence.

    """
    # Data
    posteriors: Dict[str, Any]
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
    marginalize_sigma: bool = False
    length_scales: bool = False
    
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
        event_z = self.posteriors['redshift']
        inj_z = self.injections['redshift']
        
        max_z = np.maximum(np.max(inj_z), np.max(event_z))
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

        # Interpolate and store in the dictionaries
        self.posteriors['ln_dVTc'] = jnp.interp(event_z, zs, ln_dVTc)
        self.injections['ln_dVTc'] = jnp.interp(inj_z, zs, ln_dVTc)
        
    def __post_init__(self):
        """
        Optional: Validation or automatic formatting after object creation.
        """
        if self.marginalize_sigma and self.length_scales:
            raise ValueError("Cannot marginalize over sigma with different sigmas in different axes")        
    
        # standardize bin dimension
        self.dimension = len(self.pixelpop_parameters)
        if jnp.ndim(self.bins) == 0:
            self.bins = [self.bins] * self.dimension

        self.adj_matrices = [
            create_CAR_coupling_matrix(self.bins[ii], 1, isVisible=False) for ii in range(self.dimension)
            ]

        new_minima = gwpop_models.bbh_minima.copy()
        new_maxima = gwpop_models.bbh_maxima.copy()
        
        new_minima.update(self.minima)
        new_maxima.update(self.maxima)

        self.minima = new_minima
        self.maxima = new_maxima

        # bin up events and injections
        self.event_bins, self.inj_bins, self.bin_axes, self.logdV, eprior, iprior = place_in_bins(
            self.pixelpop_parameters, 
            self.posteriors, 
            self.injections, 
            bins=self.bins, 
            minima=self.minima, 
            maxima=self.maxima
        )
        self.posteriors['log_prior'] += eprior
        self.injections['log_prior'] += iprior
        
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

        # for now, hardcode Planck15_LAL cosmology
        # TODO: allow for different cosmologies
        self.preprocess_cosmology(gwpop_models.COSMO)
