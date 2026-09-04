"""
Tests for the per-catalog default sets and the two new GWTC-6 models.

The self-consistency tests need nothing beyond pixelpop. The numerical tests
compare against ``gwtc6_population_models``, which is the authoritative source
for the GWTC-6 models, and skip if it is not installed.
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import distributions as dist

from pixelpop.models import (
    CatalogDefaults,
    GWTC_DEFAULTS,
    TriplePowerlaw_MassRatio,
    gwtc4_default,
    gwtc6_fms_default,
)

CATALOGS = sorted(GWTC_DEFAULTS)


# ---------------------------------------------------------------------------
# Self-consistency of the default sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("catalog", CATALOGS)
def test_hyperparameters_match_signatures(catalog):
    """probabilistic.py calls model(data, *[sample[h] for h in hypers]), so each
    hyperparameter list must line up *positionally* with the model signature."""
    defaults = GWTC_DEFAULTS[catalog]
    for parameter, model in defaults.models.items():
        hypers = defaults.hyperparameters.get(parameter)
        assert hypers is not None, f"{catalog}: no hyperparameters for {parameter!r}"

        signature = inspect.signature(model).parameters
        params = list(signature)[1:]  # drop `data`
        assert len(hypers) <= len(params), (
            f"{catalog}.{parameter}: {len(hypers)} hyperparameters but "
            f"{model.__name__} only takes {len(params)} arguments after `data`"
        )
        # Positional call means hypers is a *prefix* of the signature. The names
        # need not match (pixelpop renames mmin -> mlow_1, etc.), but anything
        # left off the end must be optional -- otherwise the model is called
        # with a missing required argument, or worse, everything shifts by one.
        omitted = [p for p in params[len(hypers):]
                   if signature[p].default is inspect.Parameter.empty]
        assert not omitted, (
            f"{catalog}.{parameter}: {model.__name__} requires {omitted}, "
            f"which the hyperparameter list does not supply"
        )
        assert hypers == list(dict.fromkeys(hypers)), (
            f"{catalog}.{parameter}: duplicate hyperparameter names {hypers}"
        )


@pytest.mark.parametrize("catalog", CATALOGS)
def test_every_hyperparameter_has_a_prior(catalog):
    defaults = GWTC_DEFAULTS[catalog]
    for parameter in defaults.models:
        for h in defaults.hyperparameters[parameter]:
            assert h in defaults.priors, (
                f"{catalog}: {parameter!r} needs hyperparameter {h!r}, "
                f"which has no prior in this set"
            )


@pytest.mark.parametrize("catalog", CATALOGS)
def test_priors_are_wellformed(catalog):
    for name, entry in GWTC_DEFAULTS[catalog].priors.items():
        args, distribution = entry
        assert isinstance(args, (list, tuple)), f"{catalog}.{name}: args not a sequence"
        assert issubclass(distribution, dist.Distribution), (
            f"{catalog}.{name}: {distribution!r} is not a numpyro distribution"
        )
        distribution(*args)  # must actually construct


@pytest.mark.parametrize("catalog", CATALOGS)
def test_models_evaluate_finite(catalog):
    """Every model in the set runs on a representative dataset and returns
    finite log-probabilities with finite gradients."""
    defaults = GWTC_DEFAULTS[catalog]
    rng = np.random.default_rng(0)
    n = 500
    m1 = rng.uniform(1.5, 90., n)
    data = {
        'mass_1': jnp.asarray(m1),
        'log_mass_1': jnp.asarray(np.log(m1)),
        'mass_ratio': jnp.asarray(rng.uniform(0.05, 1., n)),
        'a_1': jnp.asarray(rng.uniform(0.01, 0.99, n)),
        'a_2': jnp.asarray(rng.uniform(0.01, 0.99, n)),
        'cos_tilt_1': jnp.asarray(rng.uniform(-0.99, 0.99, n)),
        'cos_tilt_2': jnp.asarray(rng.uniform(-0.99, 0.99, n)),
        'chi_eff': jnp.asarray(rng.uniform(-0.9, 0.9, n)),
        'chi_p': jnp.asarray(rng.uniform(0.01, 0.99, n)),
        'redshift': jnp.asarray(rng.uniform(0.01, 1.8, n)),
    }
    data['mass_2'] = data['mass_1'] * data['mass_ratio']
    data['log_mass_2'] = jnp.log(data['mass_2'])

    for parameter, model in defaults.models.items():
        values = [_central_value(defaults.priors[h])
                  for h in defaults.hyperparameters[parameter]]
        out = np.asarray(model(data, *values))
        assert not np.any(np.isnan(out)), f"{catalog}.{parameter}: NaN in output"
        assert np.any(np.isfinite(out)), f"{catalog}.{parameter}: nothing finite"

        # Gradients w.r.t. the scalar hyperparameters must stay finite.
        scalar = [i for i, v in enumerate(values) if jnp.ndim(v) == 0]
        if not scalar:
            continue

        def loss(*scalars):
            full = list(values)
            for i, v in zip(scalar, scalars):
                full[i] = v
            o = model(data, *full)
            return jnp.sum(jnp.where(jnp.isfinite(o), o, 0.0))

        grads = jax.jit(jax.grad(loss, argnums=tuple(range(len(scalar)))))(
            *[values[i] for i in scalar]
        )
        for i, g in zip(scalar, grads):
            name = defaults.hyperparameters[parameter][i]
            assert np.all(np.isfinite(np.asarray(g))), (
                f"{catalog}.{parameter}: non-finite gradient w.r.t. {name}"
            )


def _central_value(prior):
    args, distribution = prior
    if distribution is dist.Delta:
        return jnp.asarray(args[0], dtype=float)
    if distribution is dist.Dirichlet:
        concentration = jnp.asarray(args[0])
        return concentration / concentration.sum()
    lo, hi = args
    return jnp.asarray((lo + hi) / 2, dtype=float)


# ---------------------------------------------------------------------------
# CatalogDefaults behaviour
# ---------------------------------------------------------------------------

def test_merge_lets_caller_win_and_leaves_original_alone():
    custom = ([-2., 10.], dist.Uniform)
    merged = gwtc6_fms_default.merge(priors={'lamb': custom})

    assert merged.priors['lamb'] == custom
    assert gwtc6_fms_default.priors['lamb'] == ([-10, 10], dist.Uniform)
    # untouched keys fall through
    assert merged.priors['mlow_1'] == gwtc6_fms_default.priors['mlow_1']
    assert merged.models is not gwtc6_fms_default.models
    assert dict(merged.models) == dict(gwtc6_fms_default.models)


def test_sets_are_frozen_and_do_not_alias_the_global_registries():
    import pixelpop.models.gwpop_models as gwpop

    with pytest.raises(Exception):
        gwtc4_default.models = {}
    with pytest.raises(TypeError):
        gwtc4_default.priors['alpha'] = None

    # gwtc4_default is built from the module-level registries but must copy them
    assert gwtc4_default.models is not gwpop.gwparameter_to_model
    assert dict(gwtc4_default.models) == dict(gwpop.gwparameter_to_model)


def test_gwtc5_is_a_distinct_object_with_the_gwtc4_models():
    assert GWTC_DEFAULTS['gwtc5'] is not gwtc4_default
    assert dict(GWTC_DEFAULTS['gwtc5'].models) == dict(gwtc4_default.models)


def test_gwtc6_fms_differs_from_gwtc6_only_where_expected():
    gwtc6 = GWTC_DEFAULTS['gwtc6']
    fms = GWTC_DEFAULTS['gwtc6_fms']

    changed = {p for p in gwtc6.models
               if gwtc6.models[p] is not fms.models.get(p)}
    assert changed == {'mass_1', 'log_mass_1', 'a', 'mass_ratio'}
    # the mass ratio resolves one slope per source class, BNS / NSBH / BBH
    assert fms.models['mass_ratio'] is TriplePowerlaw_MassRatio
    assert fms.hyperparameters['mass_ratio'] == [
        'beta_1', 'beta_2', 'beta_3', 'mlow_2', 'delta_m_2']
    for slope in ('beta_1', 'beta_2', 'beta_3'):
        assert fms.priors[slope] == ([-2, 7], dist.Uniform)
    # the full spectrum opens the minimum masses down into the NS range
    assert fms.priors['mlow_1'] == ([1., 3.], dist.Uniform)
    assert fms.priors['mlow_2'] == ([1., 3.], dist.Uniform)
    assert float(fms.minima['log_mass_1']) == pytest.approx(float(jnp.log(1.)))


def test_catalog_defaults_accepts_plain_dicts():
    d = CatalogDefaults(models={'a': lambda data, x: x},
                        hyperparameters={'a': ['x']},
                        priors={'x': ([0, 1], dist.Uniform)})
    assert d.minima == {}
    assert d.priors['x'][1] is dist.Uniform


# Numerical agreement with gwtc6_population_models now lives in gwtc6_models_test.py;
# this file covers the CatalogDefaults machinery only.
