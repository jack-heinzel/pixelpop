"""
Tests for the capped SPDE range prior, and for the prior/posterior models
building the same field.

``MaternSPDETransform`` renormalizes its spectrum to a fixed site-averaged
marginal variance, so once the range exceeds the grid it cancels out of the
likelihood entirely. The old ``Normal(0, 3)`` range prior put half its mass on
sub-pixel ranges the grid cannot represent and barely resisted the flat tail
above the domain, so runs with no short-range axis to anchor ``lnsigma``
random-walked out there and failed to converge. The prior is now
``Uniform(0, log(bins[i]))`` per axis: exactly the representable band.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.distributions.transforms import biject_to

from pixelpop.experimental.probabilistic import prior_probabilistic_model
from pixelpop.models.probabilistic import (
    get_latent_sites,
    setup_probabilistic_model,
    trace_model,
)
from pixelpop.utils.data import PixelPopData, capped_log_range_prior

BINS = [8, 5]
PARAMETERS = ['log_mass_1', 'redshift']


def _pixelpop_data(bins=BINS, **kwargs):
    """A minimal two-axis run: no parametric models, just the field."""
    rng = np.random.default_rng(0)
    nobs, npe, ninj = 4, 60, 2000
    posteriors = {
        'log_mass_1': jnp.asarray(rng.uniform(np.log(6), np.log(40), (nobs, npe))),
        'redshift': jnp.asarray(rng.uniform(0.05, 1.2, (nobs, npe))),
        'log_prior': jnp.zeros((nobs, npe)),
        'ln_dVTc': jnp.zeros((nobs, npe)),
    }
    injections = {
        'log_mass_1': jnp.asarray(rng.uniform(np.log(6), np.log(40), ninj)),
        'redshift': jnp.asarray(rng.uniform(0.05, 1.2, ninj)),
        'log_prior': jnp.zeros(ninj),
        'ln_dVTc': jnp.zeros(ninj),
        'total_generated': 1e5,
        'analysis_time': 1.0,
    }
    return PixelPopData(
        name='range_prior_test',
        posteriors=posteriors,
        injections=injections,
        pixelpop_parameters=list(PARAMETERS),
        other_parameters=[],
        bins=bins,
        minima={'log_mass_1': float(np.log(3)), 'redshift': 0.0},
        maxima={'log_mass_1': float(np.log(100)), 'redshift': 2.0},
        random_initialization=True,
        **kwargs,
        )


@pytest.fixture(scope='module')
def spde_data():
    return _pixelpop_data(spde_matern=True)


# ---------------------------------------------------------------------------
# The prior itself
# ---------------------------------------------------------------------------

def test_cap_is_the_domain_size_of_each_axis():
    (low, high), distribution = capped_log_range_prior([8, 5])
    assert distribution is dist.Uniform
    np.testing.assert_allclose(np.asarray(low), [0., 0.])
    np.testing.assert_allclose(np.asarray(high), np.log([8., 5.]), rtol=1e-6)


def test_factor_widens_the_cap():
    """factor=2 and 4 leave the residual spectrum deviation at the cap at 3% and
    1% rather than 12%, at the cost of keeping more of the flat tail."""
    (_, high), _ = capped_log_range_prior([100], factor=4.)
    np.testing.assert_allclose(np.asarray(high), np.log([400.]), rtol=1e-6)


def test_default_is_per_axis_even_for_scalar_bins():
    """`bins` may be a single int; the cap still has to be one per axis, since
    the axes need not have the same length in general."""
    pp = _pixelpop_data(bins=8, spde_matern=True)
    (low, high), distribution = pp.range_prior
    assert distribution is dist.Uniform
    assert jnp.shape(high) == (2,)
    np.testing.assert_allclose(np.asarray(high), np.log([8., 8.]), rtol=1e-6)


def test_an_explicit_prior_is_left_alone():
    """The cap is a default, not a policy: passing the old Normal(0, 3) back in
    has to keep working, including its scalar-broadcast-to-every-axis shape."""
    pp = _pixelpop_data(spde_matern=True, range_prior=((0.0, 3.0), dist.Normal))
    assert pp.range_prior == ((0.0, 3.0), dist.Normal)
    _, initial_value = setup_probabilistic_model(pp)
    assert jnp.shape(initial_value['log_ranges']) == (2,)


# ---------------------------------------------------------------------------
# What the model draws
# ---------------------------------------------------------------------------

def test_log_ranges_is_one_per_axis_and_inside_the_cap(spde_data):
    """`expand` rather than `sample_shape`: a per-axis prior handed a
    sample_shape would come back (dimension, dimension)."""
    model, initial_value = setup_probabilistic_model(spde_data)
    trace = trace_model(model, initial_value, {
        'posteriors': spde_data.posteriors, 'injections': spde_data.injections})

    log_ranges = trace['log_ranges']['value']
    assert jnp.shape(log_ranges) == (2,)
    assert (log_ranges > 0.).all()
    assert (log_ranges < jnp.log(jnp.asarray(BINS, dtype=float))).all()


def test_initial_log_ranges_is_finite_unconstrained(spde_data):
    """NUTS starts in unconstrained space, where either edge of the capped
    support is +-inf and the chain would start at nan."""
    _, initial_value = setup_probabilistic_model(spde_data)
    args, distribution = spde_data.range_prior
    unconstrained = biject_to(distribution(*args).support).inv(
        initial_value['log_ranges'])
    assert jnp.isfinite(unconstrained).all()


def test_initial_log_ranges_stays_inside_a_tiny_grid():
    """log(bins/4) is <= 0 -- outside the support -- for four bins or fewer."""
    pp = _pixelpop_data(bins=[3, 4], spde_matern=True)
    _, initial_value = setup_probabilistic_model(pp)
    assert (initial_value['log_ranges'] > 0.).all()
    assert (initial_value['log_ranges'] < jnp.log(jnp.asarray([3., 4.]))).all()


def test_every_initial_value_is_an_array(spde_data):
    """init_to_value carries these straight into the trace, and consumers that
    read .size/.shape off a trace value break on a Python scalar."""
    _, initial_value = setup_probabilistic_model(spde_data)
    assert initial_value
    for name, value in initial_value.items():
        assert isinstance(value, jnp.ndarray), name


# ---------------------------------------------------------------------------
# Prior and posterior are the same model
# ---------------------------------------------------------------------------

def _field_sites(model, initial_value, pixelpop_data):
    """The sites NUTS would adapt. trace_model conditions on the initial values,
    which marks those sites observed, so go through get_latent_sites rather than
    reading is_observed off the trace."""
    trace = trace_model(model, initial_value, {
        'posteriors': pixelpop_data.posteriors,
        'injections': pixelpop_data.injections})
    return {name: trace[name] for name in get_latent_sites(trace, initial_value)}


def test_prior_model_builds_the_same_field_as_the_posterior(spde_data):
    """The *_prior runs exist to say how far the data moved the field, which they
    can only do if both sides use the same kernel. The prior model used to be
    hard-wired to ICAR whatever spde_matern said."""
    posterior_model, posterior_init = setup_probabilistic_model(spde_data)
    prior_model, prior_init = prior_probabilistic_model(spde_data)

    posterior_sites = _field_sites(posterior_model, posterior_init, spde_data)
    prior_sites = _field_sites(prior_model, prior_init, spde_data)

    for name in ('lnsigma', 'log_nu_spde', 'log_ranges', '_eigenbasis_sites'):
        assert name in prior_sites, name
        assert (jnp.shape(prior_sites[name]['value'])
                == jnp.shape(posterior_sites[name]['value'])), name

    # ... with one deliberate exception: the prior model records the likelihood
    # but never factors it in, so a flat unbounded offset would be improper.
    assert 'log_rate_offset' in posterior_sites
    assert 'log_rate_offset' not in prior_sites


def test_prior_model_still_uses_icar_when_asked(spde_data):
    """The ICAR path stays available; the point is that it now follows the flags
    rather than being hard-wired."""
    icar_data = _pixelpop_data(diagonalize_icar=True)
    prior_model, prior_init = prior_probabilistic_model(icar_data)
    prior_sites = _field_sites(prior_model, prior_init, icar_data)
    assert 'log_ranges' not in prior_sites
    assert '_eigenbasis_sites' in prior_sites
