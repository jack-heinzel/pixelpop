import jax.numpy as jnp
import arviz as az
import xarray as xr
import warnings
import population_error

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


def compute_error_statistics(hyperposterior, pixelpop_data, verbose=True):
    """
    Compute systematic error statistics for a PixelPop inference result.

    This function calculates the information loss (in bits) due to finite 
    Monte Carlo sampling in both the single-event posterior estimation and 
    the selection function estimation. It leverages the `population_error` 
    package to compute precision (variance) and accuracy (bias) metrics. 
    For mathematical details, see https://arxiv.org/abs/2509.07221

    Parameters
    ----------
    hyperposterior : dict or pandas.DataFrame
        Samples from the population hyperposterior. Keys should match the 
        hyperparameters required by the PixelPop model (including 'merger_rate_density' 
        and any parametric hyperparameters).
    pixelpop_data : PixelPopData
        The data container used for the analysis, holding single-event posteriors, 
        injections, and configuration settings.
    verbose : bool, default=True
        Flag for printing information at runtime.

    Returns
    -------
    dict
        A dictionary containing error statistics, including:
        - 'error_statistic': Total expected information loss (bits).
        - 'precision_statistic': Information loss due to estimator variance.
        - 'accuracy_statistic': Information loss due to estimator bias.
        - 'event_precision_statistic': Variance contribution from single-event PE.
        - 'selection_precision_statistic': Variance contribution from selection effects.
        - 'event_accuracy_statistic': Bias contribution from single-event PE.
        - 'selection_accuracy_statistic': Bias contribution from selection effects.

    Notes
    -----
    This function automatically instantiates `PixelPopRateFunction` wrappers for 
    both the event posteriors and the injection set. It assumes a rate-based 
    likelihood (`rate=True`) and includes the likelihood correction term 
    (`include_likelihood_correction=True`).
    """
    if verbose:
        print('='*50)
        print('Computing error statistics')
        print('='*50 + '\n')
    
    posteriors = pixelpop_data.posteriors
    injections = pixelpop_data.injections

    posteriors['prior'] = jnp.exp(posteriors.pop('log_prior'))
    injections['prior'] = jnp.exp(injections.pop('log_prior'))
    
    # add delta parameters
    # TODO: move this to a PixelPopData function
    delta_pars = {}
    for p in pixelpop_data.other_parameters:
        for h in pixelpop_data.parameter_to_hyperparameters[p]:
            if pixelpop_data.priors[h][1].__name__ == 'Delta':
                delta_pars[h] = pixelpop_data.priors[h][0][0]

    Nsamples = len(hyperposterior['log_likelihood'])
    for par in pixelpop_data.other_parameters:
        required_keys = pixelpop_data.parameter_to_hyperparameters[par]
        for k in required_keys:
            if not k in hyperposterior:
                hyperposterior[k] = delta_pars[k]*jnp.ones(Nsamples)


    event_pixelpop_model = PixelPopRateFunction(
        pixelpop_data, dataset_type='posteriors'
    )

    injection_pixelpop_model = PixelPopRateFunction(
        pixelpop_data, dataset_type='injections'
    )

    # burn a call for each model
    first_hypersample = {k: hyperposterior[k][0] for k in hyperposterior.keys()}
    
    _ = event_pixelpop_model(posteriors, first_hypersample)
    _ = injection_pixelpop_model(injections, first_hypersample)
    
    error_dict = population_error.error_statistics(
        event_pixelpop_model, 
        injections, 
        posteriors, 
        hyperposterior, 
        vt_model_function=injection_pixelpop_model,
        include_likelihood_correction=True,
        rate=True,
        verbose=verbose,
        )
    
    return error_dict

def rank_normalized_rhat(
        hyperposterior, threshold=1.01, fail_percentage_threshold=0.01, verbose=True
        ):
    """
    Compute rank-normalized R-hat diagnostics with a high-dimensional noise filter.

    This function transforms posterior samples into an ArviZ-compatible format, 
    calculates rank-normalized R-hats, and evaluates convergence based on the 
    global distribution of R-hat values across all parameters.

    Parameters
    ----------
    hyperposterior : arviz.InferenceData
        arviz.InferenceData containing the hyperposterior samples.
    threshold : float, default 1.01
        The R-hat value above which an individual parameter is considered to have 
        "failed" convergence.
    fail_percentage_threshold : float, default 0.01
        The allowable fraction of parameters that can exceed `threshold` before 
        a warning is issued. This accounts for spurious fluctuations inherent in 
        estimating ~10^4 to ~10^6 parameters from finite samples.
    verbose : bool, default=True
        Flag for printing information at runtime.
    Returns
    -------
    rhat_results : xarray.Dataset
        An ArviZ dataset containing the R-hat values. Multi-dimensional parameters 
        retain their shape with automatically generated dimension names 
        (e.g., 'param_idx_0').
    passed : bool
        Whether the posterior satisfies the tolerance requirements for sampling 
        convergence.


    Notes
    -----
    **Rank-Normalization:**
    Uses the improved R-hat which rank-transforms samples to be robust to 
    non-Gaussianity and heavy tails. See https://arxiv.org/abs/1903.08008


    **High-Dimensional Handling:**
    For a model with 1,000,000 parameters, a 1% threshold allows for 10,000 
    parameters to "fail" by chance due to the multiple comparisons problem. 
    If the `fail_pct` is below `fail_percentage_threshold`, the high R-hats are 
    treated as sampling noise rather than structural convergence issues.

    Warnings
    --------
    warning if the percentage of parameters exceeding the threshold 
    surpasses the `fail_percentage_threshold`.
    """

    rhat_results = az.rhat(hyperposterior, method="rank")
    
    # with so many parameters, it's likely for spurious fluctuations in rhat
    # estimation with finite samples to be above threshold
    all_rhats = jnp.concatenate([rhat_results[v].values.flatten() for v in rhat_results.data_vars])
    fail_pct = (all_rhats > threshold).mean()

    if fail_pct > fail_percentage_threshold:
        if verbose:
            warnings.warn(f"Warning: {100*fail_pct:.2f}% of parameters exceed R-hat={threshold}. "
                          "This may indicate a genuine convergence failure.")
        passed = False
    else:
        if verbose:
            print(f"Convergence check: {100*fail_pct:.2f}% of parameters exceed R-hat={threshold}. "
                  "This is acceptable, and likely noise in high-dimensional estimation.")
            print(f"Mean R-hat = {all_rhats.mean()} and max R-hat = {all_rhats.max()}")
        passed = True
    return rhat_results, passed

def compute_effective_sample_sizes(
        hyperposterior, threshold=100, fail_percentage_threshold=0.01, verbose=True,
        ):
    """
    Compute Effective Sample Size (ESS) diagnostics with a high-dimensional noise filter.

    This function calculates both bulk and tail ESS. Bulk-ESS focuses on the 
    sampling efficiency of the mean, while tail-ESS focuses on the 
    efficiency of the 5% and 95% quantiles.

    Parameters
    ----------
    hyperposterior : arviz.InferenceData
        arviz.InferenceData containing the hyperposterior samples.
    threshold : float, default 100
        The ESS value below which a parameter is considered to have 
        insufficient independent samples. A good rule of thumb is 100 per chain.
    fail_percentage_threshold : float, default 0.01
        The allowable fraction of parameters that can fall below `threshold` before 
        a warning is issued.
    verbose : bool, default=True
        Flag for printing information at runtime.

    Returns
    -------
    ess_results : xarray.Dataset
        An ArviZ dataset containing the bulk and tail ESS values.
    passed : bool
        Whether the posterior satisfies the tolerance requirements for sampling 
        efficiency.

    Notes
    -----
    **Bulk vs Tail ESS:**
    Bulk-ESS is useful for diagnostics of the location of the distribution, 
    whereas Tail-ESS is useful for diagnostics of the scale/variance. 
    Low tail-ESS often indicates the sampler is not exploring the edges 
    of the high-dimensional space efficiently.

    

    Warnings
    --------
    warning if the percentage of parameters with ESS below threshold 
    surpasses the `fail_percentage_threshold`.
    """
    # Compute both types of ESS
    ess_bulk = az.ess(hyperposterior, method="bulk")
    ess_tail = az.ess(hyperposterior, method="tail")
    
    ess_results = xr.merge([
        ess_bulk.rename({v: f"{v}_bulk" for v in ess_bulk.data_vars}),
        ess_tail.rename({v: f"{v}_tail" for v in ess_tail.data_vars})
    ])

    all_ess_bulk = jnp.concatenate([ess_bulk[v].values.flatten() for v in ess_bulk.data_vars])
    fail_pct = (all_ess_bulk < threshold).mean()

    if fail_pct > fail_percentage_threshold:
        if verbose:
            warnings.warn(f"Warning: {100*fail_pct:.2f}% of parameters have a bulk ESS below {threshold}. "
                          "This indicates the sampler may not have enough independent samples for reliable inference.")
        passed = False
    else:
        if verbose:
            print(f"Efficiency check: {100*fail_pct:.2f}% of parameters have ESS below {threshold}. "
                  "Sampler efficiency appears robust.")
            print(f"Mean ESS = {all_ess_bulk.mean()} and minimum ESS = {all_ess_bulk.min()}\n")
        passed = True

    return ess_results, passed


def convert_to_arviz(hyperposterior):
    """
    Helper function for converting hyperposterior output by PixelPop 
    into an arviz InferenceData object. 

    This function standardizes the various possible formats of the 
    hyperposterior (single chain vs multiple chains) and ensures that 
    multi-dimensional parameters are labeled.

    Parameters
    ----------
    hyperposterior : dict or list of dicts
        The posterior samples to convert.
        - If a list of dicts: Each dictionary represents one chain. Arrays 
          within should have shape (n_draws, *sample_shape).
        - If a single dict: Arrays should have shape (n_chains, n_draws, *sample_shape) 
          or (n_draws, *sample_shape). The format is inferred by checking the 
          dimensionality of the 'log_likelihood' entry.

    Returns
    -------
    idata : az.InferenceData
        An ArviZ InferenceData object where the 'posterior' group contains all 
        parameters. The first two axes of every variable are mapped to 'chain' 
        and 'draw'. Internal dimensions are named '[parameter]_idx_n'.
    """
    processed_posterior = {}
    auto_dims = {}

    if isinstance(hyperposterior, list):
        keys = hyperposterior[0].keys()
        for k in keys:
            stacked = jnp.stack([chain[k] for chain in hyperposterior])
            processed_posterior[k] = stacked
            # Generate dims: ignore 0 (chain) and 1 (draw)
            if stacked.ndim > 2:
                auto_dims[k] = [f"{k}_idx_{i}" for i in range(stacked.ndim - 2)]

    elif isinstance(hyperposterior, dict):
        log_ls_ndim = hyperposterior['log_likelihood'].ndim
        for k, v in hyperposterior.items():
            if log_ls_ndim == 1:
                processed_posterior[k] = v[None, ...]
            else:
                processed_posterior[k] = v
            
            if processed_posterior[k].ndim > 2:
                auto_dims[k] = [f"{k}_idx_{i}" for i in range(processed_posterior[k].ndim - 2)]

    # ArviZ automatically maps the first two dims to 'chain' and 'draw'
    # The 'dims' dict tells it what to call everything else.
    idata = az.from_dict(
        posterior=processed_posterior,
        dims=auto_dims
    )
    return idata

def validate_pixelpop_inference(
        hyperposterior, pixelpop_data, rhat_threshold=1.01, ess_threshold=100,
        fail_percentage_threshold=0.01, verbose=True
        ):
    """
    Runs MCMC convergence (R-hat) and Monte Carlo systematics checks on PixelPop results.

    Parameters
    ----------
    hyperposterior : dict
        Posterior samples keyed by parameter name. Values should be JAX arrays 
        of shape (n_chains, n_draws, *param_shape).
    pixelpop_data : PixelPopData
        The data object containing posteriors, injections, and model settings.
    rhat_threshold : float, default=1.01
        Threshold for flagging R-hat convergence failures.
    ess_threshold : float, default=100
        Threshold for flagging effective sample size efficiency issues.
    fail_percentage_threshold : float, default=0.01
        Fraction of parameters we will tolerate failing the rhat or ess thresholds.
    verbose : bool, default=True
        Whether to pass the verbose flag to the underlying validation functions.

    Returns
    -------
    tuple
        (rhat_results, ess_results, error_stats)
    """

    # convert to arviz formatted InferenceData object
    az_posterior = convert_to_arviz(hyperposterior)

    # arviz rank normalized rhat expects an InferenceData object
    rhat_results, convergence_pass = rank_normalized_rhat(
        az_posterior, 
        threshold=rhat_threshold,
        fail_percentage_threshold=fail_percentage_threshold, 
        verbose=verbose
    )

    ess_results, efficiency_pass = compute_effective_sample_sizes(
        az_posterior, 
        threshold=ess_threshold,
        fail_percentage_threshold=fail_percentage_threshold,
        verbose=verbose
        )
    
    # convert arviz InferenceData object to dict: (samples, ...) for error stats
    flat_az_posterior = az.extract(az_posterior, combined=True)

    flat_dict_posterior = {}
    for k, v in flat_az_posterior.data_vars.items():
        # Move the last axis (samples in arviz at end) to the front 
        arr = jnp.moveaxis(v.values, -1, 0)
        flat_dict_posterior[k] = arr
    
    error_stats = compute_error_statistics(
        flat_dict_posterior, 
        pixelpop_data, 
        verbose=verbose
    )

    summary = {
        'error_statistic': error_stats['error_statistic'],
        'sampling_convergence': convergence_pass,
        'sampling_efficiency': efficiency_pass,
    }

    return rhat_results, ess_results, error_stats, summary