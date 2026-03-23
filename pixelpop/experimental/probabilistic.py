import numpy as np
from ..utils.nearest_neighbor import create_CAR_coupling_matrix
from ..models.gwpop_models import *
from ..models.car import lower_triangular_map, mult_outer, add_outer
from .car import DiagonalizedICARTransform
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.debug import print as jaxprint
from ..utils.data import place_in_bins
from jax.scipy.special import logsumexp as LSE
import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide
import jax
from jax import random
from numpyro import handlers
import os
import h5ify

def prior_probabilistic_model(pixelpop_data, log='default'):
    """
    Construct a hierarchical probabilistic model for GW population inference.

    This function sets up both parametric and nonparametric (CAR/ICAR) components
    of a gravitational-wave population model, returning a NumPyro-compatible model
    along with suitable initial values for MCMC warmup.

    Parameters
    ----------
    posteriors : dict
        Posterior samples keyed by parameter name. Each entry is shaped
        (Nobs, Nsample). Must also include 'ln_prior'.
    injections : dict
        Injection data keyed by parameter name. Each entry is shaped (Nfound).
        Must include 'ln_prior', 'total_generated' (int/float), and
        'analysis_time' (float).
    parameters : list of str
        Parameters for the nonparametric pixelized model (e.g., ["mass_1", "chi_eff"]).
    other_parameters : list of str
        Additional parameters modeled with parametric forms.
    bins : int or list of int
        Number of bins along each axis in the pixelized model.
    length_scales : bool, optional TODO!!!!
        If True, use independent CAR coupling parameters per axis.
    minima : dict, optional
        Mapping of parameter → minimum value. Defaults to typical BBH values.
    maxima : dict, optional
        Mapping of parameter → maximum value. Defaults to typical BBH values.
    parametric_models : dict, optional
        Mapping of parameter → callable defining parametric model.
    hyperparameters : dict, optional
        Mapping of parameter → list of hyperparameter names for its parametric model.
    priors : dict, optional
        Mapping of hyperparameter → (args, distribution) prior specification.
    random_initialization : bool, optional
        If True, initialize ICAR model with random noise instead of plausible values.
    lower_triangular : bool, optional
        If True, enforce p1 > p2 triangular support (used for joint m1–m2 models).
    constraint_funcs : list of callables, optional
        Extra constraint functions applied to hyperparameters.
    log : {"default", "debug"}, optional
        Logging verbosity.
        
    Returns
    -------
    probabilistic_model : callable
        NumPyro-compatible probabilistic model.
    initial_value : dict
        Suggested initial values for MCMC warmup.
    """

    posteriors = pixelpop_data.posteriors
    injections = pixelpop_data.injections
    parameters = pixelpop_data.pixelpop_parameters
    other_parameters = pixelpop_data.other_parameters
    bins = pixelpop_data.bins
    length_scales = pixelpop_data.length_scales
    minima = pixelpop_data.minima
    maxima = pixelpop_data.maxima
    parametric_models = pixelpop_data.parametric_models
    hyperparameters = pixelpop_data.parameter_to_hyperparameters
    priors = pixelpop_data.priors
    lower_triangular = pixelpop_data.lower_triangular
    constraint_funcs = pixelpop_data.constraint_funcs
    coupling_prior = pixelpop_data.coupling_prior

    dimension = len(parameters)
    if np.ndim(bins) == 0:
        bins = [bins] * dimension
    adj_matrices = [create_CAR_coupling_matrix(bins[ii], 1, isVisible=False) for ii in range(dimension)]
    
    if 'redshift' in parameters:
        from astropy.cosmology import Planck15
        from astropy import units
        max_z = np.maximum(np.max(injections['redshift']), np.max(posteriors['redshift']))
        zs = np.linspace(1e-6, max_z, 10000)
        dVs = Planck15.differential_comoving_volume(zs) * 4 * np.pi * units.sr
        ln_dVTc = np.log(dVs.to(units.Gpc**3).value) - np.log(1 + zs)
        event_z = posteriors['redshift']
        inj_z = injections['redshift']
        event_ln_dVTc = jnp.array(np.interp(event_z, zs, ln_dVTc))
        inj_ln_dVTc = jnp.array(np.interp(inj_z, zs, ln_dVTc))
    else:
        event_ln_dVTc = jnp.zeros_like(posteriors['log_prior'])
        inj_ln_dVTc = jnp.zeros_like(injections['log_prior'])
        
    event_bins, inj_bins, bin_axes, logdV, eprior, iprior = place_in_bins(parameters, posteriors, injections, bins=bins, minima=minima, maxima=maxima)
    
    posteriors['log_prior'] += eprior
    injections['log_prior'] += iprior

    # update models
    parameter_to_hyperparameters = gwparameter_to_hyperparameters.copy()
    parameter_to_hyperparameters.update(hyperparameters)

    parameter_to_gwpop_model = {}
    for p in other_parameters:
        if p in parametric_models:
            print(f'Updating {p} model from {gwparameter_to_model[p].__name__} to {parametric_models[p].__name__}')
            print(f'\t ...with hyperparameters {parameter_to_hyperparameters[p]}')
            parameter_to_gwpop_model[p] = parametric_models[p]
        else:
            print(f'Using default {p} model {gwparameter_to_model[p].__name__}')
            parameter_to_gwpop_model[p] = gwparameter_to_model[p]

    # default priors
    hyperparameter_priors = {}
    for p in other_parameters:
        for h in parameter_to_hyperparameters[p]:
            if h in priors:    
                pprint = priors[h]
                print(f'Using custom prior {h} = {pprint[1].__name__}({str(pprint[0])[1:-1]}) in {p} model')
                hyperparameter_priors[h] = priors[h]
            else:
                pprint = default_priors[h]
                print(f'Using default prior {h} = {pprint[1].__name__}({str(pprint[0])[1:-1]}) in {p} model')
                hyperparameter_priors[h] = default_priors[h]

    if lower_triangular:
        lt_map = lower_triangular_map(bins[0])
        tri_size = int(bins[0]*(bins[0]+1)/2) 
        unique_sample_shape = (tri_size,) + tuple(bins[2:])
        # lower triangular in first two dimensions
    else:
        unique_sample_shape = bins
    normalization_dof = np.prod(unique_sample_shape)

    def get_initial_value():
        """
        Construct initial values for the pixelized (nonparametric) merger rate density.

        Parameters
        ----------
        plausible_hyperparameters : dict
            Plausible hyperparameter values used for initialization if not random.
        parameters : list of str
            Parameters included in the nonparametric model.
        Nobs : int
            Number of observed events.
        inj_weights : ndarray
            Logarithmic weights from injections, adjusted for prior volume.
        random_initialization : bool
            If True, initialize randomly; otherwise use plausible hyperparameters.

        Returns
        -------
        initial_value : dict
            Dictionary containing initial 'merger_rate_density' or 'base_interpolation'.
        """
        return_dict = {'_eigenbasis_sites': jnp.array(
            np.random.normal(loc=0, scale=1, size=unique_sample_shape))
            }
        return return_dict
            
    initial_value = get_initial_value()

    def parametric_prior(data, injections, event_weights, inj_weights):
        """
        Evaluate the parametric population model contribution.

        Draws hyperparameters from their priors
        """
        sample = {}
        for key in hyperparameter_priors:
            args, distribution = hyperparameter_priors[key]
            if distribution.__name__ == 'Delta':
                sample[key] = args[0]
            else:
                sample[key] = numpyro.sample(key, distribution(*args))        
        for constraint_func in constraint_funcs:
            numpyro.factor(constraint_func.__name__, constraint_func(sample))
            if log == 'debug':
                jaxprint('[DEBUG] constraint functions:', constraint_func.__name__, constraint_func(sample))

        for p in other_parameters:
            event_weights += parameter_to_gwpop_model[p](data, *[sample[h] for h in parameter_to_hyperparameters[p]])
            inj_weights += parameter_to_gwpop_model[p](injections, *[sample[h] for h in parameter_to_hyperparameters[p]])
            if log == 'debug':
                jaxprint('[DEBUG] parametric {p} LSE(event_weights)={ew}, LSE(injection_weights)={iw}', p=p, ew=LSE(event_weights), iw=LSE(inj_weights))
                if not jnp.isfinite(LSE(event_weights)):
                    for parameter in map_to_gwpop_parameters[p]:
                        jaxprint('[DEBUG] inf event weights at {p}={d}', p=parameter, d=data[parameter][jnp.where(event_weights == jnp.inf)])
                if not jnp.isfinite(LSE(inj_weights)):
                    for parameter in map_to_gwpop_parameters[p]:
                        jaxprint('[DEBUG] inf injection weights at {p}={d}', p=parameter, d=injections[parameter][jnp.where(inj_weights == jnp.inf)])
        return event_weights, inj_weights
    def nonparametric_prior(event_bins, inj_bins, event_weights, inj_weights):
        """
        Evaluate the nonparametric (ICAR/CAR) pixelized model contribution.
        """
        if length_scales:
            # TODO!!
            lsigma = numpyro.sample('lnsigma', coupling_prior[1](*coupling_prior[0]), sample_shape=(dimension,))
        else:
            lsigma = numpyro.sample('lnsigma', coupling_prior[1](*coupling_prior[0]), sample_shape=()) 

        mask = jnp.ones(bins, dtype=bool).at[(0,) * len(unique_sample_shape)].set(False)

        _eigenbasis_sites = numpyro.sample(
            "_eigenbasis_sites",
            dist.Normal(0., 1.).expand(unique_sample_shape).mask(mask)
        )
        # _eigenbasis_site_0 = numpyro.sample("_eigenbasis_site_0", dist.ImproperUniform(dist.constraints.real, (), ()))
        _eigenbasis_site_0 = 0.
        eigenbasis_sites = _eigenbasis_sites.at[(0,) * dimension].set(_eigenbasis_site_0)
        
        if lower_triangular:
            eigenbasis_sites = lower_triangular_map(eigenbasis_sites)

        merger_rate_density = numpyro.deterministic(
            'merger_rate_density',
            DiagonalizedICARTransform(lsigma, adj_matrices, is_sparse=True)(
                eigenbasis_sites
            )
        )
        if not lower_triangular:
            normalization = numpyro.deterministic('log_rate', LSE(merger_rate_density)+jnp.sum(logdV))
            for ii, p in enumerate(parameters):
                sum_axes = tuple(np.arange(dimension)[np.r_[0:ii,ii+1:dimension]])
                numpyro.deterministic(f'log_marginal_{p}', LSE(merger_rate_density-normalization, axis=sum_axes) + jnp.sum(logdV[:ii]) + jnp.sum(logdV[ii+1:]))

        event_weights += merger_rate_density[event_bins] # (69,3194)
        inj_weights += merger_rate_density[inj_bins]
        if log == 'debug':
            jaxprint('[DEBUG] pixelpop LSE(event_weights)={ew}, LSE(injection_weights)={iw}', ew=LSE(event_weights), iw=LSE(inj_weights))
        return event_weights, inj_weights
        
    def probabilistic_model(posteriors, injections):
        """
        Full probabilistic model for hierarchical GW population inference.

        Combines the nonparametric pixelized rate density with parametric models,
        applies detection efficiency corrections, and evaluates the likelihood.

        Parameters
        ----------
        posteriors : dict
            Posterior samples from detected events.
        injections : dict
            Injection data including selection effects.

        Side Effects
        ------------
        Stores deterministic nodes in NumPyro for logging:
        - log_likelihood
        - log_likelihood_variance
        - pe_variance
        - vt_variance
        - Nexp

        Returns
        -------
        None
            (Factors likelihood into NumPyro’s computation graph.)
        """
        event_weights, inj_weights = nonparametric_prior(event_bins, inj_bins, event_ln_dVTc-posteriors['log_prior'], inj_ln_dVTc-injections['log_prior'])
        event_weights, inj_weights = parametric_prior(posteriors, injections, event_weights, inj_weights)

        ln_likelihood, nexp, pe_var, vt_var, total_var = rate_likelihood(event_weights, inj_weights, injections['total_generated'], live_time=injections['analysis_time'])

        # save these values!
        numpyro.deterministic("log_likelihood", ln_likelihood)
        numpyro.deterministic("log_likelihood_variance", total_var)
        numpyro.deterministic("pe_variance", pe_var)
        numpyro.deterministic("vt_variance", vt_var)
        numpyro.deterministic("Nexp", nexp)

    return probabilistic_model, initial_value


def _run_svi_chunk(svi, svi_state, num_steps, stable_update):
    """Run a chunk of SVI steps using lax.scan for speed."""
    def _step(state, _):
        state, loss = svi.stable_update(state) if stable_update else svi.update(state)
        return state, loss
    svi_state, losses = jax.lax.scan(_step, svi_state, None, length=num_steps)
    return svi_state, losses


def svi_inference(
    probabilistic_model, model_kwargs={}, initial_value={},
    guide_type='mvn', num_steps=50000, lr=1e-3, num_samples=1000,
    rng_key=random.PRNGKey(1), run_dir='./', name='',
    stable_update=False, num_particles=1, progress_bar=True,
    max_patience=None, smoothing_window=1000, parallel=1,
    chain_offset=0, num_flows=2, hidden_dims=None, hidden_factors=None,
):
    """
    Approximate the pixelpop posterior using NumPyro's stochastic variational
    inference. Mirrors the interface of ``inference_loop``.

    Parameters
    ----------
    probabilistic_model : callable
        NumPyro probabilistic model (as returned by ``setup_probabilistic_model``).
    model_kwargs : dict, optional
        Keyword arguments passed to the probabilistic model (e.g.,
        ``{'posteriors': ..., 'injections': ...}``).
    initial_value : dict, optional
        Initial parameter values used to seed the guide's location parameters.
    guide_type : {'diagonal', 'mvn', 'iaf', 'bnaf'}, optional
        Variational family for the guide:
        - ``'diagonal'``: mean-field Normal (``AutoNormal``).
        - ``'mvn'``: full-rank multivariate Normal (``AutoMultivariateNormal``).
        - ``'iaf'``: inverse autoregressive flow (``AutoIAFNormal``).
        - ``'bnaf'``: block neural autoregressive flow (``AutoBNAFNormal``).
        Default ``'mvn'``.
    num_steps : int, optional
        Maximum number of SVI optimization steps (default 50000).
    lr : float, optional
        Adam learning rate (default 1e-3).
    num_samples : int, optional
        Number of posterior samples to draw from each fitted guide (default 1000).
    rng_key : jax.random.PRNGKey, optional
        Random key for reproducibility (default ``PRNGKey(1)``).
    run_dir : str, optional
        Directory to save output samples (default ``'./'``).
    name : str, optional
        Subdirectory name for this run.
    stable_update : bool, optional
        If ``True``, clip large gradient updates during optimization, which can
        help with tricky posteriors (default ``False``).
    num_particles : int, optional
        Number of particles (samples) used to estimate the ELBO at each step
        (default 1). Increasing this reduces gradient variance at the cost of
        speed.
    progress_bar : bool, optional
        Show a progress bar during optimization (default ``True``).
    max_patience : int or None, optional
        If set, stop early when the smoothed ELBO has not improved for
        ``max_patience`` chunks. If ``None`` (default), run for the full
        ``num_steps``.
    smoothing_window : int, optional
        Chunk size (in steps) for evaluating patience (default 1000).
        Ignored if ``max_patience`` is ``None``.
    parallel : int or list, optional
        Number of independent SVI runs (default 1). Each run re-initializes
        the optimizer from scratch with a different RNG key. If a list of ints,
        specifies the chain indices used in save filenames.
    chain_offset : int, optional
        Offset applied to chain index when saving outputs (default 0).
    num_flows : int, optional
        Number of flow transformations for ``'iaf'`` and ``'bnaf'`` guides
        (default 2). Ignored for ``'diagonal'`` and ``'mvn'``.
    hidden_dims : list of int or None, optional
        Hidden layer sizes for each flow's neural network in ``'iaf'``
        guides. If ``None`` (default), uses numpyro's default of
        ``[latent_dim, latent_dim]``. Each layer must be at least as wide
        as the latent dimension due to the autoregressive masking
        structure. Ignored for other guide types.
    hidden_factors : list of int or None, optional
        Hidden layer sizes for ``'bnaf'`` guides, expressed as multipliers
        of the latent dimension. If ``None`` (default), uses ``[1]``.
        Numpyro's built-in default of ``[8, 8]`` is usually far too large.
        Ignored for other guide types.

    Returns
    -------
    samples : list of dict
        List of posterior sample dicts, one per chain.
    svi_results : list of SVIRunResult
        List of SVI results containing ``params`` and ``losses``.
    guides : list of AutoGuide
        List of fitted guide objects.
    """
    import optax
    from numpyro.infer.svi import SVIRunResult

    # Wrap model so data kwargs are baked in (SVI needs a zero-argument model)
    def conditioned_model():
        probabilistic_model(**model_kwargs)

    # Select guide family
    guide_map = {
        'diagonal': autoguide.AutoNormal,
        'mvn': autoguide.AutoMultivariateNormal,
        'iaf': autoguide.AutoIAFNormal,
        'bnaf': autoguide.AutoBNAFNormal,
    }
    if guide_type not in guide_map:
        raise ValueError(
            f"Unknown guide_type '{guide_type}'. Choose from {list(guide_map.keys())}."
        )

    if not isinstance(parallel, (list, tuple)):
        parallel = list(range(parallel))
    rng_keys = random.split(rng_key, num=len(parallel))

    all_samples = []
    all_svi_results = []
    all_guides = []

    for ii, chain in enumerate(parallel):
        print(f"SVI chain #{chain} out of {parallel}")
        chain_rng = rng_keys[ii]
        chain_rng, run_key, sample_key = random.split(chain_rng, 3)

        # Fresh guide and optimizer for each chain
        guide_kwargs = dict(
            init_loc_fn=numpyro.infer.init_to_value(values=initial_value),
        )
        if guide_type == 'iaf':
            guide_kwargs['num_flows'] = num_flows
            if hidden_dims is not None:
                guide_kwargs['hidden_dims'] = hidden_dims
        elif guide_type == 'bnaf':
            guide_kwargs['num_flows'] = num_flows
            guide_kwargs['hidden_factors'] = hidden_factors if hidden_factors is not None else [1]
        guide = guide_map[guide_type](conditioned_model, **guide_kwargs)
        optimizer = numpyro.optim.optax_to_numpyro(optax.adam(lr))
        loss_fn = Trace_ELBO(num_particles=num_particles)
        svi = SVI(conditioned_model, guide, optimizer, loss=loss_fn)

        if max_patience is None:
            svi_result = svi.run(
                run_key,
                num_steps,
                stable_update=stable_update,
                progress_bar=progress_bar,
            )
        else:
            chunk_size = smoothing_window
            num_chunks = int(np.ceil(num_steps / chunk_size))
            chunk_losses_list = []
            best_mean_loss = np.inf
            best_params = None
            chunks_without_improvement = 0
            svi_state = svi.init(run_key)

            for chunk in range(num_chunks):
                svi_state, chunk_losses = _run_svi_chunk(svi, svi_state, chunk_size, stable_update)
                chunk_losses = np.array(chunk_losses)
                chunk_losses_list.append(chunk_losses)

                mean_loss = float(np.mean(chunk_losses))
                total_steps = (chunk + 1) * chunk_size
                if mean_loss < best_mean_loss:
                    best_mean_loss = mean_loss
                    best_params = svi.get_params(svi_state)
                    chunks_without_improvement = 0
                else:
                    chunks_without_improvement += 1

                if progress_bar:
                    print(f"\r[chain {chain} SVI step {total_steps}/{num_steps}] "
                          f"mean loss: {mean_loss:.2f}, "
                          f"best: {best_mean_loss:.2f}, "
                          f"patience: {chunks_without_improvement}/{max_patience}",
                          end="", flush=True)

                if chunks_without_improvement >= max_patience:
                    print(f"\nEarly stopping at step {total_steps} "
                          f"(no improvement for {max_patience} chunks "
                          f"of {chunk_size} steps)")
                    break

            if chunks_without_improvement < max_patience and progress_bar:
                print()
            all_chunk_losses = jnp.concatenate(chunk_losses_list)
            svi_result = SVIRunResult(best_params, svi_state, all_chunk_losses)

        # Discover all site names by tracing the guide then replaying the model.
        # We can't trace the model directly because some distributions (e.g. ICAR)
        # don't implement sample().
        guide_trace_fn = handlers.substitute(guide, svi_result.params)
        with handlers.seed(rng_seed=0):
            guide_trace = handlers.trace(guide_trace_fn).get_trace()
        guide_sites = [
            name for name, site in guide_trace.items()
            if site["type"] == "sample"
        ]
        guide_values = {name: site["value"] for name, site in guide_trace.items()
                        if site["type"] == "sample"}
        replayed_model = handlers.substitute(conditioned_model, guide_values)
        with handlers.seed(rng_seed=0):
            model_trace = handlers.trace(replayed_model).get_trace()
        det_sites = [
            name for name, site in model_trace.items()
            if site["type"] == "deterministic"
        ]
        all_sites = guide_sites + det_sites

        # Draw posterior samples one at a time via lax.map to avoid OOM
        predictive = numpyro.infer.Predictive(
            conditioned_model,
            guide=guide,
            params=svi_result.params,
            num_samples=1,
            return_sites=all_sites,
        )

        def _draw_one(key):
            return predictive(key)

        sample_keys = random.split(sample_key, num_samples)
        stacked = jax.lax.map(_draw_one, sample_keys)
        samples = {k: v.squeeze(1) for k, v in stacked.items()}

        # Save results
        os.makedirs(os.path.join(run_dir, name), exist_ok=True)
        f = os.path.join(run_dir, name, f'svi_{chain+chain_offset}_samples.h5')
        h5ify.save(f, {k: np.array(v) for k, v in samples.items()}, mode='w')
        np.save(os.path.join(run_dir, name, f'svi_{chain+chain_offset}_losses.npy'),
                np.array(svi_result.losses))

        all_samples.append(samples)
        all_svi_results.append(svi_result)
        all_guides.append(guide)
    if len(parallel) == 1:
        return samples, svi_result, guide
    return all_samples, all_svi_results, all_guides


def neutra_inference(
    probabilistic_model, guide, svi_params, model_kwargs={},
    warmup=5000, tot_samples=100, thinning=100, pacc=0.65,
    maxtreedepth=10, num_samples=1, parallel=1,
    rng_key=random.PRNGKey(1), cache_cadence=1, run_dir='./', name='',
    print_keys=['Nexp', 'log_likelihood', 'log_likelihood_variance'],
    chain_offset=0, dense_mass=False,
):
    """
    Run NUTS in the NeuTra-reparameterized space of a fitted SVI guide.

    Uses :class:`~numpyro.infer.reparam.NeuTraReparam` to transform the model
    so that NUTS samples in the guide's approximately unit-Gaussian latent
    space, then transforms samples back to the original parameterization.

    Requires a pre-fitted SVI guide (e.g. from ``svi_inference``).
    Mirrors the interface of ``inference_loop``.

    Parameters
    ----------
    probabilistic_model : callable
        NumPyro probabilistic model (as returned by ``setup_probabilistic_model``).
    guide : AutoGuide
        A fitted ``AutoContinuous`` guide (e.g. from ``svi_inference``).
    svi_params : dict
        Optimized variational parameters from SVI (``svi_result.params``).
    model_kwargs : dict, optional
        Keyword arguments passed to the probabilistic model.
    warmup : int, optional
        Number of warmup iterations (default 5000).
    tot_samples : int, optional
        Total number of posterior samples to save per chain.
    thinning : int, optional
        Interval between recorded samples (default 100).
    pacc : float, optional
        Target acceptance probability for NUTS (default 0.65).
    maxtreedepth : int, optional
        Maximum tree depth for NUTS (default 10).
    num_samples : int, optional
        Number of samples per thinning block (default 1).
    parallel : int or list, optional
        Number of independent chains (default 1). If a list of ints,
        specifies the chain indices used in save filenames.
    rng_key : jax.random.PRNGKey, optional
        Random key for reproducibility.
    cache_cadence : int, optional
        Interval between checkpoint saves (default 1).
    run_dir : str, optional
        Directory to save output chains (default ``'./'``).
    name : str, optional
        Subdirectory name for this run.
    print_keys : list of str, optional
        Keys to include in periodic summaries.
    chain_offset : int, optional
        Offset applied to chain index when saving outputs (default 0).
    dense_mass : bool, optional
        Whether to use a dense mass matrix in NUTS (default False).

    Returns
    -------
    samples : list of dict
        List of posterior samples for each chain.
    mcmc : numpyro.infer.MCMC
        Completed MCMC sampler instance.
    """
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.reparam import NeuTraReparam
    from numpyro.diagnostics import print_summary
    from ..models.probabilistic import get_worst_rhat_neff
    from tqdm import tqdm
    from contextlib import redirect_stdout
    import sys

    neutra = NeuTraReparam(guide, svi_params)
    reparam_model = neutra.reparam(probabilistic_model)

    skip_keys = [k[1:] for k in print_keys if k.startswith('~')]

    # Compute table size from print_keys by tracing the reparam'd model.
    # Use handlers.seed so sample sites that lack .sample() get seeded values.
    # Condition on guide samples to avoid ICAR NotImplementedError.
    guide_trace_fn = handlers.substitute(guide, svi_params)
    with handlers.seed(rng_seed=0):
        _gt = handlers.trace(guide_trace_fn).get_trace()
    _guide_vals = {n: s["value"] for n, s in _gt.items() if s["type"] == "sample"}
    _cond_model = handlers.condition(reparam_model, data=_guide_vals)
    with handlers.seed(rng_seed=0):
        _trace = handlers.trace(_cond_model).get_trace(**model_kwargs)
    table_size = 2
    for pk in print_keys:
        if pk.startswith('~'):
            continue
        if pk in _trace:
            table_size += _trace[pk]["value"].size

    # Initialize at the origin of the NeuTra latent space, which maps
    # to the guide's mean in the original parameterization.
    latent_name = f"{guide.prefix}_shared_latent"
    latent_dim = guide.get_base_dist().shape()[-1]
    neutra_init = {latent_name: jnp.zeros(latent_dim)}

    kernel = NUTS(
        reparam_model,
        max_tree_depth=maxtreedepth,
        target_accept_prob=pacc,
        dense_mass=dense_mass,
        init_strategy=numpyro.infer.init_to_value(values=neutra_init),
    )

    if not isinstance(parallel, (list, tuple)):
        parallel = list(range(parallel))
    rng_keys = random.split(rng_key, num=len(parallel))

    samples = []
    for ii, chain in enumerate(parallel):
        chain_key = rng_keys[ii]
        print(f"NeuTra chain #{chain} out of {parallel}")
        mcmc = MCMC(
            kernel,
            thinning=thinning,
            num_warmup=warmup,
            num_samples=num_samples * thinning,
            num_chains=1,
        )

        mcmc.warmup(chain_key, **model_kwargs)
        sys.stdout.write("\n" * (table_size + 3))
        chain_samples = None
        mcmc.transfer_states_to_host()
        sample_iterator = tqdm(range(int(1e-4 + tot_samples / num_samples)))
        sample_iterator.set_description("drawing thinned samples")
        for sample in sample_iterator:
            mcmc.post_warmup_state = mcmc.last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, **model_kwargs)
            next_sample = mcmc.get_samples()
            sys.stdout.write("\x1b[1A\n\x1b[1A")

            if chain_samples is None:
                chain_samples = {key: np.array(next_sample[key]) for key in next_sample}
            else:
                for key in chain_samples:
                    chain_samples[key] = np.concatenate(
                        (chain_samples[key], np.array(next_sample[key])), axis=0
                    )
            mcmc.transfer_states_to_host()
            key0 = list(chain_samples.keys())[0]
            if (sample % cache_cadence == 0) and (chain_samples[key0].shape[0] >= 4):
                sys.stdout.write(f"\x1b[1A\x1b[2K" * (table_size + 3))

                rhat_key, rhat_chain, neff_key, neff_chain = get_worst_rhat_neff(
                    chain_samples, skip_keys=skip_keys
                )
                summary_dict = {
                    key: chain_samples[key]
                    for key in print_keys
                    if key[1:] not in skip_keys and key in chain_samples
                }
                summary_dict['worst r_hat: ' + rhat_key] = rhat_chain
                summary_dict['worst n_eff: ' + neff_key] = neff_chain

                print_summary(summary_dict, group_by_chain=False)
                os.makedirs(os.path.join(run_dir, name), exist_ok=True)
                with open(os.path.join(run_dir, name, f'neutra_{chain+chain_offset}_metadata.txt'), 'w+') as f_meta:
                    with redirect_stdout(f_meta):
                        print_summary(summary_dict, group_by_chain=False)
                f = os.path.join(run_dir, name, f'neutra_{chain+chain_offset}_samples.h5')
                h5ify.save(f, chain_samples, mode='w')

        samples.append(chain_samples)

    return samples, mcmc