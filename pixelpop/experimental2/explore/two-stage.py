def evaluate_stage1_parametric_on_pp_grid(pixelpop_data, stage1_samples, point_estimate="median"):
    """
    Build the frozen Stage-1 baseline mu_grid = log p_param(theta_PP) on the PixelPop grid.

    Notes
    -----
    - Uses pixelpop_data.mixture_parametric_models for the PixelPop parameters.
    - Builds a full grid_data dict for ALL pixelpop_parameters, so conditional
      models like p(chi_eff | z) can evaluate correctly.
    - Returns mu_grid in log-space (up to an additive constant).
    """
    bins = tuple(int(b) for b in pixelpop_data.bins)
    dimension = int(pixelpop_data.dimension)

    # ---------- choose point estimates ----------
    fixed_hypers = {}
    if point_estimate == "map":
        if "log_likelihood" not in stage1_samples:
            raise ValueError("point_estimate='map' requires 'log_likelihood' in stage1_samples")
        ll = np.asarray(stage1_samples["log_likelihood"]).reshape(-1)
        imax = int(np.argmax(ll))
    else:
        imax = None  # median mode

    for key, val in stage1_samples.items():
        arr = np.asarray(val)
        if arr.ndim == 0:
            fixed_hypers[key] = float(arr)
            continue
        arr = arr.reshape(-1)
        if arr.size == 0:
            continue
        if point_estimate == "median":
            fixed_hypers[key] = float(np.median(arr))
        elif point_estimate == "map":
            fixed_hypers[key] = float(arr[imax])
        else:
            raise ValueError(f"Unknown point_estimate: {point_estimate}")

    # ---------- build grid coordinates for all PP dimensions ----------
    # bin_axes may be centers (len=bins) OR edges (len=bins+1), depending on PixelPopData internals.
    grid_axes = []
    if hasattr(pixelpop_data, "bin_axes") and pixelpop_data.bin_axes is not None:
        for ii in range(dimension):
            ax = np.asarray(pixelpop_data.bin_axes[ii])

            # If bin_axes are edges, convert to centers
            if ax.shape[0] == bins[ii] + 1:
                ax = 0.5 * (ax[:-1] + ax[1:])
            elif ax.shape[0] == bins[ii]:
                pass  # already centers
            else:
                raise ValueError(
                    f"Unexpected bin axis length for dim {ii}: got {ax.shape[0]}, "
                    f"expected {bins[ii]} (centers) or {bins[ii]+1} (edges)"
                )

            grid_axes.append(ax)
    else:
        # fallback: reconstruct centers from minima/maxima
        for ii, p in enumerate(pixelpop_data.pixelpop_parameters):
            pmin = float(pixelpop_data.minima[p])
            pmax = float(pixelpop_data.maxima[p])
            edges = np.linspace(pmin, pmax, bins[ii] + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            grid_axes.append(centers)

    grid_data = {}
    for ii, p in enumerate(pixelpop_data.pixelpop_parameters):
        centers = jnp.asarray(grid_axes[ii])
        shape = [1] * dimension
        shape[ii] = bins[ii]
        grid_data[p] = centers.reshape(shape)

    # ---------- sum the parametric PP log-models ----------
    mu_grid = jnp.zeros(bins)
    for p in pixelpop_data.pixelpop_parameters:
        model_fn = pixelpop_data.mixture_parametric_models[p]
        hypers = [fixed_hypers[h] for h in pixelpop_data.mixture_parameter_to_hyperparameters[p]]
        # model_fn returns log-pdf contribution
        mu_grid = mu_grid + model_fn(grid_data, *hypers)

    return mu_grid, fixed_hypers


def build_stage1_other_prior_overrides(pixelpop_data, stage1_samples, mode="fixed", point_estimate="median"):
    """
    Build prior overrides for OTHER-parameter hyperparameters only (not PP hyperparameters).

    Parameters
    ----------
    mode : "fixed" or "gaussian"
        fixed   -> Delta prior at Stage-1 point estimate
        gaussian-> Normal prior using Stage-1 mean/std
    """
    from numpyro.distributions import Delta, Normal

    other_hyper_keys = set()
    for p in pixelpop_data.other_parameters:
        for h in pixelpop_data.parameter_to_hyperparameters[p]:
            other_hyper_keys.add(h)

    skip_keys = {
        # common deterministic outputs / diagnostics
        "log_likelihood", "log_likelihood_variance", "pe_variance", "vt_variance", "Nexp",
        "log_rate", "interpolant", "interpolant_normalization", "unscaled_gamma", "lnsigma",
        "deviation_field", "deviation_rms", "deviation_max_abs", "parametric_baseline",
    }

    overrides = {}

    if point_estimate == "map":
        if "log_likelihood" not in stage1_samples:
            raise ValueError("point_estimate='map' requires 'log_likelihood' in stage1_samples")
        ll = np.asarray(stage1_samples["log_likelihood"]).reshape(-1)
        imax = int(np.argmax(ll))
    else:
        imax = None

    for key, val in stage1_samples.items():
        if key in skip_keys:
            continue
        if key not in other_hyper_keys:
            continue

        arr = np.asarray(val).reshape(-1)
        if arr.size < 1:
            continue

        if mode == "fixed":
            if point_estimate == "median":
                x0 = float(np.median(arr))
            elif point_estimate == "map":
                x0 = float(arr[imax])
            else:
                raise ValueError(f"Unknown point_estimate: {point_estimate}")
            overrides[key] = ([x0], Delta)

        elif mode == "gaussian":
            mu = float(np.mean(arr))
            sig = float(np.std(arr))
            sig = max(sig, 1e-8)
            overrides[key] = ([mu, sig], Normal)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return overrides


def setup_twostage_informed_probabilistic_model(
    pixelpop_data,
    mu_grid,
    other_prior_overrides=None,
    log="default",
):
    """
    Stage-2 informed residual PixelPop model for GW population inference.

    Model
    -----
    log R(theta) = log_rate
                 + log p_other(theta_other)
                 + mu_grid(theta_PP)
                 + delta(theta_PP)

    where:
      - mu_grid is frozen from Stage 1 (parametric baseline on PP subspace)
      - delta is an ICAR residual field (sigma marginalized ICAR prior)
    """

    if pixelpop_data.lower_triangular:
        raise NotImplementedError("Lower triangular not yet supported in twostage informed model.")

    bins = tuple(int(b) for b in pixelpop_data.bins)
    dimension = int(pixelpop_data.dimension)
    n_pix = int(np.prod(bins))

    mu_grid = jnp.asarray(mu_grid)
    if tuple(mu_grid.shape) != bins:
        raise ValueError(f"mu_grid shape {mu_grid.shape} does not match bins {bins}")

    # Merge priors (allow stage-1-informed overrides for OTHER params only)
    all_priors = dict(pixelpop_data.priors)
    if other_prior_overrides is not None:
        all_priors.update(other_prior_overrides)

    # ----------------------------------------------------------------
    # Initial values
    # ----------------------------------------------------------------
    # Pin one pixel of delta to zero -> sample only n_pix-1 free parameters
    # Small random init avoids quad=0 edge cases
    delta0 = 1e-3 * np.random.normal(size=(n_pix - 1,))
    initial_value = {
        "delta_free": jnp.asarray(delta0),
        "log_rate": jnp.log(50.0),
    }

    # ----------------------------------------------------------------
    # Parametric model for OTHER parameters only
    # ----------------------------------------------------------------
    def parametric_model_other(data, injections):
        sample = {}

        other_event = 0.0
        other_inj = 0.0

        # draw/supply OTHER hyperparameters (and any constraints they need)
        for key, (args, distribution) in all_priors.items():
            # Only sample params that actually appear in the "other" models
            # (this avoids sampling the PP hyperparameters, which are baked into mu_grid)
            is_other_hyper = False
            for p in pixelpop_data.other_parameters:
                if key in pixelpop_data.parameter_to_hyperparameters[p]:
                    is_other_hyper = True
                    break
            if not is_other_hyper:
                continue

            if distribution.__name__ == "Delta":
                sample[key] = args[0]
            else:
                sample[key] = numpyro.sample(key, distribution(*args))

        for constraint_func in pixelpop_data.constraint_funcs:
            numpyro.factor(constraint_func.__name__, constraint_func(sample))

        for p in pixelpop_data.other_parameters:
            model_fn = pixelpop_data.parametric_models[p]
            hypers = [sample[h] for h in pixelpop_data.parameter_to_hyperparameters[p]]

            other_event = other_event + model_fn(data, *hypers)
            other_inj = other_inj + model_fn(injections, *hypers)

            if log == "debug":
                jaxprint(
                    "[DEBUG] other {p}: LSE(event)={ew}, LSE(inj)={iw}",
                    p=p, ew=LSE(other_event), iw=LSE(other_inj)
                )

        return other_event, other_inj

    # ----------------------------------------------------------------
    # Residual PixelPop model (mu + delta)
    # ----------------------------------------------------------------
    def residual_pp_model(event_bins, inj_bins):
        log_rate = numpyro.sample(
            "log_rate",
            dist.ImproperUniform(dist.constraints.real, (), ())
        )

        # Gauge-fix delta by pinning one pixel to zero
        n_free = n_pix - 1
        delta_free = numpyro.sample(
            "delta_free",
            dist.ImproperUniform(dist.constraints.real_vector, (), (n_free,))
        )
        delta_flat = jnp.concatenate([jnp.zeros((1,)), delta_free], axis=0)
        delta_field = jnp.reshape(delta_flat, bins)

        # Sigma-marginalized ICAR prior on delta
        ICAR_sigma_marg = initialize_sigma_marginalized_ICAR(dimension)
        icar_prior = ICAR_sigma_marg(
            single_dimension_adj_matrices=pixelpop_data.adj_matrices,
            is_sparse=True,
        )

        icar_lp, quad = icar_prior.log_prob_and_quad(delta_field)
        numpyro.factor("icar_deviation_prior", icar_lp)

        # Optional implied sigma diagnostic
        unscaled_gamma = numpyro.sample(
            "unscaled_gamma",
            numpyro.distributions.Gamma(concentration=(n_pix / 2.0))
        )
        precision = unscaled_gamma * quad / 2.0
        numpyro.deterministic("lnsigma", -0.5 * jnp.log(precision))

        interpolant = mu_grid + delta_field  # full PP log-shape (unnormalized, okay for GW)

        # Diagnostics
        numpyro.deterministic("parametric_baseline", mu_grid)
        numpyro.deterministic("deviation_field", delta_field)
        numpyro.deterministic("interpolant", interpolant)
        numpyro.deterministic("deviation_rms", jnp.sqrt(jnp.mean(delta_field**2)))
        numpyro.deterministic("deviation_max_abs", jnp.max(jnp.abs(delta_field)))

        # Log marginals for baseline and final (PP subspace)
        for ii, p in enumerate(pixelpop_data.pixelpop_parameters):
            sum_axes = tuple(np.arange(dimension)[np.r_[0:ii, ii+1:dimension]])

            numpyro.deterministic(
                f"log_marginal_{p}_baseline",
                LSE(mu_grid, axis=sum_axes)
                + jnp.sum(pixelpop_data.logdV[:ii])
                + jnp.sum(pixelpop_data.logdV[ii+1:])
            )

            numpyro.deterministic(
                f"log_marginal_{p}",
                LSE(interpolant, axis=sum_axes)
                + jnp.sum(pixelpop_data.logdV[:ii])
                + jnp.sum(pixelpop_data.logdV[ii+1:])
            )

            numpyro.deterministic(
                f"mean_deviation_{p}",
                jnp.mean(delta_field, axis=sum_axes)
            )

        event_pp = interpolant[event_bins]
        inj_pp = interpolant[inj_bins]

        if log == "debug":
            jaxprint(
                "[DEBUG] residual PP: delta_rms={rms}, max|delta|={mx}",
                rms=jnp.sqrt(jnp.mean(delta_field**2)),
                mx=jnp.max(jnp.abs(delta_field))
            )

        return log_rate, event_pp, inj_pp

    # ----------------------------------------------------------------
    # Full GW rate model
    # ----------------------------------------------------------------
    def probabilistic_model(posteriors, injections):
        base_event = posteriors["ln_dVTc"] - posteriors["log_prior"]
        base_inj = injections["ln_dVTc"] - injections["log_prior"]

        other_event, other_inj = parametric_model_other(posteriors, injections)
        log_rate, event_pp, inj_pp = residual_pp_model(pixelpop_data.event_bins, pixelpop_data.inj_bins)

        event_weights = base_event + log_rate + other_event + event_pp
        inj_weights = base_inj + log_rate + other_inj + inj_pp

        ln_likelihood, nexp, pe_var, vt_var, total_var = rate_likelihood(
            event_weights,
            inj_weights,
            injections["total_generated"],
            live_time=injections["analysis_time"],
        )

        steep = 0.1
        taper = smooth(total_var, pixelpop_data.UncertaintyCut ** 2, steep)

        numpyro.deterministic("log_likelihood", ln_likelihood)
        numpyro.deterministic("log_likelihood_variance", total_var)
        numpyro.deterministic("pe_variance", pe_var)
        numpyro.deterministic("vt_variance", vt_var)
        numpyro.deterministic("Nexp", nexp)

        numpyro.factor("log_likelihood_plus_taper", ln_likelihood + taper)

    return probabilistic_model, initial_value