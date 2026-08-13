import jax.numpy as jnp
import arviz as az
import numpy as np
import xarray as xr
import warnings
import population_error
from .post_processing import PixelPopRateFunction


def _as_dataset(obj):
    """Coerce an arviz rhat/ess result to a Dataset (arviz>=1 returns a DataTree)."""
    if isinstance(obj, xr.Dataset):
        return obj
    to_dataset = getattr(obj, "to_dataset", None)
    if to_dataset is not None:
        return to_dataset()
    return obj


def compute_error_statistics(hyperposterior, pixelpop_data, verbose=True):
    """
    Compute systematic error statistics for a PixelPop inference result.

    Information loss (in bits) due to finite Monte Carlo sampling in the
    single-event posterior and selection function estimates, via the
    `population_error` package. See https://arxiv.org/abs/2509.07221

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
    """
    if verbose:
        print('='*50)
        print('Computing error statistics')
        print('='*50 + '\n')
    
    posteriors = pixelpop_data.posteriors
    injections = pixelpop_data.injections

    posteriors['prior'] = jnp.exp(posteriors.get('log_prior'))
    injections['prior'] = jnp.exp(injections.get('log_prior'))
    
    # add delta parameters
    hyperposterior, Nsamples = pixelpop_data.fill_out_hyperposterior(hyperposterior)

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
    
    # Posteriors are a rectangular (Nobs, NPE) dict; events with fewer real samples are
    # padded with prior=+inf rows (zero weight). event_counts gives each event's real
    # sample count so the single-event Monte-Carlo integral size is correct.
    error_dict = population_error.error_statistics(
        event_pixelpop_model,
        injections,
        posteriors,
        hyperposterior,
        vt_model_function=injection_pixelpop_model,
        include_likelihood_correction=True,
        rate=True,
        verbose=verbose,
        event_counts=pixelpop_data.event_counts,
        )
    
    return error_dict

def rank_normalized_rhat(
        hyperposterior, threshold=1.01, fail_percentage_threshold=0.01, verbose=True
        ):
    """
    Compute rank-normalized R-hat diagnostics with a high-dimensional noise filter.

    Rank-normalized R-hat is robust to non-Gaussianity and heavy tails; see
    https://arxiv.org/abs/1903.08008

    Parameters
    ----------
    hyperposterior : arviz.InferenceData
        arviz.InferenceData containing the hyperposterior samples.
    threshold : float, default 1.01
        The R-hat value above which an individual parameter is considered to have 
        "failed" convergence.
    fail_percentage_threshold : float, default 0.01
        The allowable fraction of parameters that can exceed `threshold` before
        a warning is issued, accounting for multiple comparisons across the
        ~10^4 to ~10^6 parameters of the field.
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

    Warnings
    --------
    Warns if the percentage of parameters exceeding the threshold surpasses
    `fail_percentage_threshold`.
    """
    if verbose:
        print('='*50)
        print('Computing rank-normalized rhats')
        print('='*50 + '\n')

    rhat_results = _as_dataset(az.rhat(hyperposterior, method="rank"))

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

    Warnings
    --------
    Warns if the percentage of parameters with a bulk ESS below `threshold`
    surpasses `fail_percentage_threshold`.
    """
    if verbose:
        print('='*50)
        print('Computing aggregate effective sample sizes')
        print('='*50 + '\n')
    ess_bulk = _as_dataset(az.ess(hyperposterior, method="bulk"))
    ess_tail = _as_dataset(az.ess(hyperposterior, method="tail"))

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
            stacked = np.stack([chain[k] for chain in hyperposterior])
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

    # arviz>=1 takes a nested mapping; arviz<1 used the `posterior=` kwarg.
    if int(az.__version__.split('.')[0]) >= 1:
        idata = az.from_dict({"posterior": processed_posterior}, dims=auto_dims)
    else:
        idata = az.from_dict(posterior=processed_posterior, dims=auto_dims)
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

    # convert arviz InferenceData object to dict: (samples, ...) for error stats
    flat_az_posterior = az.extract(az_posterior, combined=True)

    flat_dict_posterior = {}
    for k, v in flat_az_posterior.data_vars.items():
        # Move the last axis (samples in arviz at end) to the front 
        arr = np.moveaxis(v.values, -1, 0)
        flat_dict_posterior[k] = arr
    
    error_stats = compute_error_statistics(
        flat_dict_posterior, 
        pixelpop_data, 
        verbose=verbose
    )

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

    summary = {
        'error_statistic': error_stats['error_statistic'],
        'sampling_convergence': convergence_pass,
        'sampling_efficiency': efficiency_pass,
    }

    return rhat_results, ess_results, error_stats, summary
