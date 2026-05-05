import numpy as np
from ..utils.nearest_neighbor import create_CAR_coupling_matrix
from ..models.gwpop_models import * 
from ..models.car import lower_triangular_map, mult_outer, add_outer
from .car import (DiagonalizedICARTransform, initialize_ICAR_normalized)
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.debug import print as jaxprint
from ..utils.data import place_in_bins
from jax.scipy.special import logsumexp as LSE
import numpyro


def setup_mixture_probabilistic_model(pixelpop_data, log="default"):
    """
    Construct a mixture probabilistic model for GW population inference.

    Closely follows the structure of probabilistic-first-attempt.py, but
    the PixelPop field is now a **normalized probability density**
    (using ICAR_normalized) rather than a log-rate-density.

    Returns
    -------
    probabilistic_model : callable
        NumPyro model function.
    initial_value : dict
        Initial values for MCMC warmup.
    """

    if pixelpop_data.lower_triangular:
        raise NotImplementedError(
            "Lower triangular not yet supported in mixture model."
        )

    normalization_dof = int(np.prod(pixelpop_data.bins))

    # ----------------------------------------------------------------
    # Build the ICAR_normalized instance
    # Same class as in the first attempt and toy notebook, but using
    # the normalized version (n-1 DOF) so the field is a proper density.
    # ----------------------------------------------------------------
    ICAR_model = initialize_ICAR_normalized(pixelpop_data.dimension)

    # ----------------------------------------------------------------
    # Initial values
    # ----------------------------------------------------------------
    if pixelpop_data.skip_nonparametric:
        initial_value = {}
    else:
        # Random initialization with small noise around zero.
        # After ICAR_normalized's constraint, this maps to roughly
        # uniform density over the pixel grid.
        initial_value = {
            "merger_rate_density": jnp.array(
                np.random.normal(
                    loc=0, scale=1, size=tuple(pixelpop_data.bins)
                )
            ),
            "log_rate": jnp.log(50.0),
        }

    # ----------------------------------------------------------------
    # Parametric model: draws ALL hyperparameters, returns weights
    # split into "common" (chi_eff, z) and "strong" (m1, q)
    # ----------------------------------------------------------------
    def parametric_model(data, injections):
        """
        Draws all hyperparameters and evaluates parametric models.

        Returns
        -------
        common_param_event_weights, common_param_inj_weights :
            Log-weights from common parameters (chi_eff, z, etc.)
        strong_param_event_weights, strong_param_inj_weights :
            Log-weights from mixture parameters (m1, q, etc.)
        """
        sample = {}

        common_param_event_weights = 0
        common_param_inj_weights = 0
        strong_param_event_weights = 0
        strong_param_inj_weights = 0

        # Draw ALL hyperparameters from priors (common + pixelpop)
        for key in pixelpop_data.priors:
            args, distribution = pixelpop_data.priors[key]
            if distribution.__name__ == "Delta":
                sample[key] = args[0]
            else:
                sample[key] = numpyro.sample(key, distribution(*args))

        # Draw mixture-specific hyperparameters (not already drawn)
        for key in pixelpop_data.mixture_priors:
            if key not in sample:
                args, distribution = pixelpop_data.mixture_priors[key]
                if distribution.__name__ == "Delta":
                    sample[key] = args[0]
                else:
                    sample[key] = numpyro.sample(key, distribution(*args))

        if log == "debug":
            for p in pixelpop_data.other_parameters + pixelpop_data.pixelpop_parameters:
                jaxprint("[DEBUG] =================================")
                jaxprint("[DEBUG] parametric parameters: {p}", p=p)
                for k in pixelpop_data.parameter_to_hyperparameters[p]:
                    if k in sample:
                        jaxprint("[DEBUG] \t {k} sample: {s}", k=k, s=sample[k])

        for constraint_func in pixelpop_data.constraint_funcs:
            numpyro.factor(constraint_func.__name__, constraint_func(sample))

        # Common parameters (multiply entire mixture: chi_eff, redshift, etc.)
        for p in pixelpop_data.common_strong_parameters:
            model_fn = pixelpop_data.parametric_models[p]
            hypers = [
                sample[h] for h in pixelpop_data.parameter_to_hyperparameters[p]
            ]
            common_param_event_weights += model_fn(data, *hypers)
            common_param_inj_weights += model_fn(injections, *hypers)

            if log == "debug":
                jaxprint(
                    "[DEBUG] common param {p}: LSE(ew)={ew}, LSE(iw)={iw}",
                    p=p,
                    ew=LSE(common_param_event_weights),
                    iw=LSE(common_param_inj_weights),
                )

        # Strong parameters (mixture side: m1, q, etc.)
        for p in pixelpop_data.mixture_parameters:
            model_fn = pixelpop_data.mixture_parametric_models[p]
            hypers = [
                sample[h]
                for h in pixelpop_data.mixture_parameter_to_hyperparameters[p]
            ]
            strong_param_event_weights += model_fn(data, *hypers)
            strong_param_inj_weights += model_fn(injections, *hypers)

            if log == "debug":
                jaxprint(
                    "[DEBUG] strong param {p}: LSE(ew)={ew}, LSE(iw)={iw}",
                    p=p,
                    ew=LSE(strong_param_event_weights),
                    iw=LSE(strong_param_inj_weights),
                )

        return (
            common_param_event_weights,
            common_param_inj_weights,
            strong_param_event_weights,
            strong_param_inj_weights,
        )

    # ----------------------------------------------------------------
    # Nonparametric model: normalized ICAR
    # ----------------------------------------------------------------
    def nonparametric_model(event_bins, inj_bins, skip=False):
        """
        Evaluate the nonparametric PixelPop contribution.

        Uses the ICAR_normalized class with ImproperUniform + factor,
        identical to the first-attempt pattern but with a NORMALIZED
        field (log-probability-density instead of log-rate-density).

        Returns
        -------
        event_weights_PP, inj_weights_PP :
            Log p_PP evaluated at event and injection bin locations.
        """
        if skip:
            log_R0 = numpyro.sample(
                "log_rate",
                dist.ImproperUniform(dist.constraints.real, (), ()),
            )
            return log_R0[None, None], log_R0[None]

        # --- Separate overall rate ---
        log_R0 = numpyro.sample(
            "log_rate",
            dist.ImproperUniform(dist.constraints.real, (), ()),
        )

        # --- Normalized ICAR field ---
        icar = ICAR_model(
            single_dimension_adj_matrices=pixelpop_data.adj_matrices,
            is_sparse=True,
            dx=jnp.exp(jnp.sum(pixelpop_data.logdV)),
        )

        # Sample unconstrained field, evaluate ICAR prior as factor
        merger_rate_density = numpyro.sample(
            "merger_rate_density",
            dist.ImproperUniform(
                dist.constraints.real, tuple(pixelpop_data.bins), ()
            ),
        )
        prior_factor, quad = icar.log_prob_and_quad(merger_rate_density)
        numpyro.factor("prior_factor", prior_factor)

        # --- Diagnostics ---
        # Normalization check: for the normalized ICAR, the field should
        # satisfy LSE(field) + log(dV) ≈ 0 when on the constraint surface.
        # Since we use ImproperUniform (not constrained sampling), this
        # is NOT enforced and will drift. The ICAR prior penalizes
        # deviations through the quadratic form.
        interpolant_normalization = LSE(merger_rate_density) + jnp.sum(
            pixelpop_data.logdV
        )
        numpyro.deterministic(
            "interpolant_normalization", interpolant_normalization
        )

        # Log marginals for diagnostic plots
        for ii, p in enumerate(pixelpop_data.pixelpop_parameters):
            sum_axes = tuple(
                np.arange(pixelpop_data.dimension)[
                    np.r_[0:ii, ii + 1 : pixelpop_data.dimension]
                ]
            )
            numpyro.deterministic(
                f"log_marginal_{p}",
                LSE(merger_rate_density, axis=sum_axes)
                + jnp.sum(pixelpop_data.logdV[:ii])
                + jnp.sum(pixelpop_data.logdV[ii + 1 :]),
            )

        # Recover sigma from conditional Gamma
        unscaled_gamma = numpyro.sample(
            "unscaled_gamma",
            numpyro.distributions.Gamma(
                concentration=((normalization_dof - 1) / 2)
            ),
        )
        precision = unscaled_gamma * quad / 2
        numpyro.deterministic("lnsigma", -0.5 * jnp.log(precision))

        # Return log p_PP at event and injection locations
        event_weights_PP = merger_rate_density[event_bins]
        inj_weights_PP = merger_rate_density[inj_bins]

        if log == "debug":
            jaxprint(
                "[DEBUG] pixelpop LSE(ew_PP)={ew}, LSE(iw_PP)={iw}",
                ew=LSE(event_weights_PP),
                iw=LSE(inj_weights_PP),
            )

        return log_R0, event_weights_PP, inj_weights_PP

    # ----------------------------------------------------------------
    # Full probabilistic model
    # ----------------------------------------------------------------
    def probabilistic_model(posteriors, injections):
        """
        Full mixture probabilistic model.

        Structure:
            event_weights = ln_dVTc - log_prior
                          + common_parametric (chi_eff, z)
                          + log_R0
                          + log[ξ · p_PP(m1,q) + (1-ξ) · p_param(m1) · p_param(q)]
        """
        # Base weights: comoving volume factor / PE prior
        base_event = posteriors["ln_dVTc"] - posteriors["log_prior"]
        base_inj = injections["ln_dVTc"] - injections["log_prior"]

        # Parametric models: common (chi_eff, z) and strong (m1, q)
        (
            common_param_event_weights,
            common_param_inj_weights,
            strong_param_event_weights,
            strong_param_inj_weights,
        ) = parametric_model(posteriors, injections)

        # --- Case 1: skip PixelPop entirely (purely parametric) ---
        if pixelpop_data.skip_nonparametric:
            log_R0 = numpyro.sample(
                "log_rate",
                dist.ImproperUniform(dist.constraints.real, (), ()),
            )
            event_weights = (
                base_event
                + log_R0
                + common_param_event_weights
                + strong_param_event_weights
            )
            inj_weights = (
                base_inj
                + log_R0
                + common_param_inj_weights
                + strong_param_inj_weights
            )

        # --- Case 2: PixelPop only (no mixture) ---
        elif pixelpop_data.skip_mixture:
            log_R0, event_weights_PP, inj_weights_PP = nonparametric_model(
                pixelpop_data.event_bins,
                pixelpop_data.inj_bins,
                skip=False,
            )
            event_weights = (
                base_event
                + log_R0
                + event_weights_PP
                + common_param_event_weights
            )
            inj_weights = (
                base_inj
                + log_R0
                + inj_weights_PP
                + common_param_inj_weights
            )

        # --- Case 3: Full mixture ---
        else:
            log_R0, event_weights_PP, inj_weights_PP = nonparametric_model(
                pixelpop_data.event_bins,
                pixelpop_data.inj_bins,
                skip=False,
            )

            xi_args, xi_dist = pixelpop_data.xi_prior
            xi = numpyro.sample("xi", xi_dist(*xi_args))

            # Both sides are now log-probability-densities, so the
            # logaddexp mixture is well-defined:
            #   log[ξ · p_PP + (1-ξ) · p_param]
            event_weights = (
                base_event
                + common_param_event_weights
                + log_R0
                + jnp.logaddexp(
                    jnp.log(xi) + event_weights_PP,
                    jnp.log1p(-xi) + strong_param_event_weights,
                )
            )
            inj_weights = (
                base_inj
                + common_param_inj_weights
                + log_R0
                + jnp.logaddexp(
                    jnp.log(xi) + inj_weights_PP,
                    jnp.log1p(-xi) + strong_param_inj_weights,
                )
            )

        # --- Poisson rate likelihood ---
        ln_likelihood, nexp, pe_var, vt_var, total_var = rate_likelihood(
            event_weights,
            inj_weights,
            injections["total_generated"],
            live_time=injections["analysis_time"],
        )

        taper = smooth(total_var, pixelpop_data.UncertaintyCut ** 2, 0.1)

        # Diagnostics
        numpyro.deterministic("log_likelihood", ln_likelihood)
        numpyro.deterministic("log_likelihood_variance", total_var)
        numpyro.deterministic("pe_variance", pe_var)
        numpyro.deterministic("vt_variance", vt_var)
        numpyro.deterministic("Nexp", nexp)

        numpyro.factor("log_likelihood_plus_taper", ln_likelihood + taper)

    return probabilistic_model, initial_value