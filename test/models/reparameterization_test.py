"""
Tests for the ordered-hyperparameter reparameterization.

``mlow_1`` and ``mlow_2`` used to be independent ``Uniform(lo, hi)`` sites with the
ordering imposed afterwards as a penalty factor, leaving the unordered half of the
square flat in the log density -- no gradient, so a chain that wandered in could
sit there for the rest of the run. They are now drawn as a pair uniform on the
triangle ``{lo <= mlow_2 <= mlow_1 <= hi}``, ordered by construction.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.distributions.transforms import biject_to
from numpyro.infer import Predictive

from pixelpop.models.gwtc_defaults import (
    gwtc3_default,
    gwtc6_default,
    gwtc6_fms_default,
)
from pixelpop.models.reparameterization import (
    ORDERED_PAIR_SUFFIX,
    ordered_pair_bounds,
    ordered_pair_initial_value,
    ordered_pair_site,
    reparameterized_sites,
    sample_ordered_pair,
)

# The pair is drawn through one length-2 site, index 0 the `mlow_2` fraction and
# index 1 the `mlow_1` fraction.
PAIR_SITE = ordered_pair_site(('mlow_1', 'mlow_2'))

# (lo, hi) for the two catalogs that sample the pair: BBH and full mass spectrum.
BOUNDS = [(3., 10.), (1., 3.)]


def _pair_model(lo, hi):
    sample_ordered_pair('mlow_1', 'mlow_2', lo, hi)


def _draw(lo, hi, num_samples=200_000, seed=0):
    samples = Predictive(_pair_model, num_samples=num_samples)(
        jax.random.PRNGKey(seed), lo, hi
    )
    return np.asarray(samples['mlow_1']), np.asarray(samples['mlow_2'])


# ---------------------------------------------------------------------------
# Which runs reparameterize
# ---------------------------------------------------------------------------

def test_gwtc6_catalogs_reparameterize_at_their_own_bounds():
    """The bounds come from the priors, not a hardcoded [3, 10]: the full mass
    spectrum has to reach the neutron stars, so its floor is 1 Msun."""
    assert ordered_pair_bounds(gwtc6_default.priors) == {('mlow_1', 'mlow_2'): (3., 10.)}
    assert ordered_pair_bounds(gwtc6_fms_default.priors) == {('mlow_1', 'mlow_2'): (1., 3.)}


def test_catalogs_without_the_pair_are_untouched():
    """GWTC-3 has a single mmin, so there is nothing to order."""
    assert ordered_pair_bounds(gwtc3_default.priors) == {}
    assert reparameterized_sites(gwtc3_default.priors) == {}
    assert ordered_pair_initial_value(gwtc3_default.priors) == {}


def test_a_fixed_member_disables_the_reparameterization():
    priors = dict(gwtc6_default.priors)
    priors['mlow_2'] = ([3.], dist.Delta)
    assert ordered_pair_bounds(priors) == {}


def test_mismatched_ranges_raise():
    priors = dict(gwtc6_default.priors)
    priors['mlow_2'] = ([3, 8], dist.Uniform)
    with pytest.raises(ValueError, match='share a prior range'):
        ordered_pair_bounds(priors)


def test_both_members_map_to_the_one_vector_site():
    """The pair is one site, not two: stacked over a chain it is (nsamples, 2), so
    the ndim == 1 filter that picks the published hyperparameters out of the
    hyperposterior drops it without having to know its name."""
    assert PAIR_SITE == 'mlow' + ORDERED_PAIR_SUFFIX
    assert reparameterized_sites(gwtc6_default.priors) == {
        'mlow_1': PAIR_SITE,
        'mlow_2': PAIR_SITE,
    }


def test_an_unregistered_pair_has_no_site():
    with pytest.raises(KeyError, match='ordered hyperparameter pair'):
        ordered_pair_site(('mmax_1', 'mmax_2'))


# ---------------------------------------------------------------------------
# The prior the reparameterization induces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('lo, hi', BOUNDS)
def test_every_draw_is_ordered_and_in_range(lo, hi):
    mlow_1, mlow_2 = _draw(lo, hi)
    assert (mlow_2 <= mlow_1).all()
    assert (mlow_2 >= lo).all() and (mlow_1 <= hi).all()


@pytest.mark.parametrize('lo, hi', BOUNDS)
def test_pair_is_uniform_on_the_triangle(lo, hi):
    """Uniform on {lo <= mlow_2 <= mlow_1 <= hi} means P(mlow_1 <= lo + t(hi-lo))
    is the area ratio t^2, and the means sit at the triangle's centroid."""
    mlow_1, mlow_2 = _draw(lo, hi)
    for t in (0.25, 0.5, 0.75):
        assert float((mlow_1 <= lo + t * (hi - lo)).mean()) == pytest.approx(t ** 2, abs=2e-3)
    assert mlow_2.mean() == pytest.approx(lo + (hi - lo) / 3, abs=1e-2)
    assert mlow_1.mean() == pytest.approx(lo + 2 * (hi - lo) / 3, abs=1e-2)


def test_the_pair_is_not_pushed_against_the_upper_bound():
    """Beta(2, 1) spans the same triangle with the wrong density on it, piling
    mlow_2 up near hi -- the corner where low-secondary-mass events fall below the
    mass-ratio cutoff and chains used to stall. Uniform-on-triangle puts a quarter
    of the prior above 6.58 Msun (the smallest per-event max m2 among the events
    that dropped out); Beta(2, 1) puts three quarters there."""
    _, mlow_2 = _draw(3., 10.)
    assert float((mlow_2 > 6.58).mean()) == pytest.approx(0.239, abs=5e-3)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('defaults', [gwtc6_default, gwtc6_fms_default])
def test_initial_value_sits_just_inside_the_low_corner(defaults):
    (lo, hi), = ordered_pair_bounds(defaults.priors).values()
    initial_value = ordered_pair_initial_value(defaults.priors)

    frac = initial_value[PAIR_SITE]
    mlow_2 = lo + (hi - lo) * frac[0]
    mlow_1 = mlow_2 + (hi - mlow_2) * frac[1]
    assert mlow_2 == pytest.approx(lo + 0.01)
    assert mlow_1 == pytest.approx(lo + 0.02)


@pytest.mark.parametrize('defaults', [gwtc6_default, gwtc6_fms_default])
def test_initial_value_is_finite_in_unconstrained_space(defaults):
    """NUTS works in unconstrained space, where the exact corner is logit(0) =
    -inf and the chain would start at nan."""
    initial_value = ordered_pair_initial_value(defaults.priors)
    unconstrained = biject_to(dist.Beta(1., 1.).support).inv(
        initial_value[PAIR_SITE]
    )
    assert jnp.isfinite(unconstrained).all()
    assert (unconstrained < 0.).all()  # still hard against the corner


def test_plausible_hyperparameters_override_the_corner():
    initial_value = ordered_pair_initial_value(
        gwtc6_default.priors, {'mlow_1': 7., 'mlow_2': 5.}
    )
    frac = initial_value[PAIR_SITE]
    assert 3. + 7. * frac[0] == pytest.approx(5.)
    assert 5. + 5. * frac[1] == pytest.approx(7.)


def test_out_of_triangle_plausible_values_are_pulled_back_inside():
    """An unordered or out-of-range pair would otherwise give a fraction outside
    (0, 1), which is not a point NUTS can start from."""
    initial_value = ordered_pair_initial_value(
        gwtc6_default.priors, {'mlow_1': 2., 'mlow_2': 40.}
    )
    for value in initial_value.values():
        assert ((0. < value) & (value < 1.)).all()


# ---------------------------------------------------------------------------
# Site bookkeeping
# ---------------------------------------------------------------------------

def test_mlow_sites_survive_as_deterministics():
    """fill_out_hyperposterior, the popsummary samples and the reweighting all read
    mlow_1/mlow_2 by name, so both have to keep appearing in the samples."""
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(_pair_model, rng_seed=0)
    ).get_trace(3., 10.)

    assert trace['mlow_1']['type'] == 'deterministic'
    assert trace['mlow_2']['type'] == 'deterministic'
    assert trace[PAIR_SITE]['type'] == 'sample'
    assert trace[PAIR_SITE]['value'].shape == (2,)


def test_the_pair_can_be_summarized_from_its_initial_value():
    """init_to_value carries the initial value straight into the trace, so a
    Python float here reaches get_table_size as trace[name]['value'].size and
    raises AttributeError before the first sample is drawn."""
    from pixelpop.models.probabilistic import get_table_size

    initial_value = ordered_pair_initial_value(gwtc6_default.priors)
    assert isinstance(initial_value[PAIR_SITE], jnp.ndarray)
    size = get_table_size(
        lambda: sample_ordered_pair('mlow_1', 'mlow_2', 3., 10.),
        initial_value, {}, ['mlow_1', 'mlow_2'],
        )
    assert size == 2 + 1 + 1  # the two header rows, plus a row per scalar site
