"""
Tests for the low-mass turn-on, :func:`pixelpop.models.base_models.m_smoother`.

The taper is exact over the turn-on itself; what these check is the two ends,
where it cannot be evaluated directly. Below the edge it used to clip to a flat
``-delta/buffer`` -- no gradient in the mass, the edge or the width, so a sampler
with an event below the minimum mass had nothing telling it which way was out. It
now floors at a fixed depth and keeps falling.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pixelpop.models.base_models import (
    BELOW_EDGE_SLOPE,
    EDGE_FRACTION,
    m_smoother,
)

# (minimum, delta). The last two are the mass-ratio window, where the variable
# runs over [0, 1] rather than over solar masses.
CASES = [(5., 3.), (5., 1.), (3., 0.5), (1.1, 5.), (8.876, 3.05), (0.1, 0.3)]

# Values and gradients are checked over this whole sweep, edges included.
SWEEP = np.concatenate([np.linspace(-5., 20., 2001), [0., 1.1, 3., 5., 8.876]])


def exact_planck_taper(m, minimum, delta):
    """Eq. (B5) of arXiv:2111.03634, in float64 numpy, unclipped."""
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    out = np.full(m.shape, -np.inf)
    out[m >= minimum + delta] = 0.
    inside = (m > minimum) & (m < minimum + delta)
    m_prime = m[inside] - minimum
    out[inside] = -np.logaddexp(0., delta/m_prime + delta/(m_prime - delta))
    return out


def _grads(m, minimum, delta):
    """d/dm, d/dminimum, d/ddelta of the summed log taper."""
    grad = jax.grad(lambda mm, mn, d: m_smoother(mm, mn, d).sum(), argnums=(0, 1, 2))
    return [np.asarray(g) for g in grad(jnp.asarray(m), minimum, delta)]


# ---------------------------------------------------------------------------
# The taper itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('minimum, delta', CASES)
def test_matches_the_exact_taper_across_the_turn_on(minimum, delta):
    """Everything above the clip junction is the textbook taper, untouched.

    Relative, not absolute: the junction end of the range sits near -1/EDGE_FRACTION,
    where float32 alone is worth ~1e-4 absolute."""
    m = minimum + np.linspace(EDGE_FRACTION, 1.5, 500) * delta
    got = np.asarray(m_smoother(jnp.asarray(m), minimum, delta), dtype=np.float64)
    np.testing.assert_allclose(got, exact_planck_taper(m, minimum, delta),
                               rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('minimum, delta', CASES)
def test_saturates_at_one_above_the_turn_on(minimum, delta):
    above = jnp.asarray([minimum + 1.01 * delta, minimum + 5. * delta])
    assert np.allclose(np.asarray(m_smoother(above, minimum, delta)), 0., atol=1e-6)


# ---------------------------------------------------------------------------
# Below the edge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('minimum, delta', CASES)
def test_floor_depth_does_not_scale_with_the_turn_on_width(minimum, delta):
    """The old clip bottomed out at -delta/buffer: a -1/buffer gradient pushing
    delta to zero for no physical reason. Clipping a fraction of delta instead
    floors at -1/EDGE_FRACTION whatever the width is."""
    at_edge = float(m_smoother(jnp.asarray([minimum]), minimum, delta)[0])
    assert at_edge == pytest.approx(-1. / EDGE_FRACTION, rel=2e-3)


@pytest.mark.parametrize('minimum, delta', CASES)
def test_below_the_edge_is_not_flat(minimum, delta):
    """The point of the change: every gradient a sampler could follow back out of
    the region is nonzero, and points the right way."""
    m = jnp.asarray([minimum - 0.5 * delta, minimum - 2. * delta])
    d_dm, d_dminimum, d_ddelta = _grads(m, minimum, delta)

    assert np.all(d_dm > 0.)          # raising the mass helps
    assert d_dminimum < 0.            # raising the edge hurts
    assert d_ddelta > 0.              # a wider turn-on leaks further down
    assert np.allclose(d_dm, BELOW_EDGE_SLOPE / delta)


@pytest.mark.parametrize('minimum, delta', CASES)
def test_keeps_falling_with_distance_below_the_edge(minimum, delta):
    deficits = np.array([0., 0.5, 1., 3.]) * delta
    values = np.asarray(m_smoother(jnp.asarray(minimum - deficits), minimum, delta))
    assert np.all(np.diff(values) < 0.)
    # ... at the documented rate, in units of the turn-on width
    assert values[-1] == pytest.approx(values[0] - 3. * BELOW_EDGE_SLOPE, rel=1e-3)


@pytest.mark.parametrize('minimum, delta', CASES)
def test_monotone_and_continuous_through_the_junction(minimum, delta):
    m = np.linspace(minimum - 2. * delta, minimum + 1.5 * delta, 20001)
    values = np.asarray(m_smoother(jnp.asarray(m), minimum, delta), dtype=np.float64)
    assert np.all(np.diff(values) >= -1e-4)
    # no cliff: the taper is steep just above the junction but never discontinuous
    below_junction = m < minimum + EDGE_FRACTION * delta
    assert np.max(np.abs(np.diff(values[below_junction]))) < 1.


# ---------------------------------------------------------------------------
# Nothing is nan, anywhere
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('delta', [0., 1e-12, 1e-8, 1e-4, 1e-2, 0.3, 1., 3., 10.])
@pytest.mark.parametrize('minimum', [0.1, 1.1, 5., 8.876])
def test_value_and_gradients_are_finite_everywhere(minimum, delta):
    """Including at delta = 0, where the taper is 0/0 and a single `where` would
    leave a nan in the gradient of the branch that is taken."""
    assert np.all(np.isfinite(np.asarray(m_smoother(jnp.asarray(SWEEP), minimum, delta))))
    for gradient in _grads(SWEEP, minimum, delta):
        assert np.all(np.isfinite(gradient))


def test_zero_width_is_a_step_continuous_with_the_narrow_limit():
    """delta = 0 takes its own branch; it should not be a cliff away from the
    narrowest width that does not."""
    minimum = 5.
    step = np.asarray(m_smoother(jnp.asarray([4.5, 5.5]), minimum, 0.))
    narrow = np.asarray(m_smoother(jnp.asarray([4.5, 5.5]), minimum, 1e-9))
    assert step[1] == pytest.approx(0.) and narrow[1] == pytest.approx(0.)
    assert step[0] < -1. / EDGE_FRACTION
    assert step[0] == pytest.approx(narrow[0], rel=1e-3)
