"""
Tests for the window factors ``create_popsummary`` folds into the pixelated rate.

A window multiplies the merger rate by a fixed shape, so on the grid it is just
``w`` evaluated at the bin centres. The case worth testing is the other one: a
window that depends on a parameter which was never pixelated -- the secondary
mass of a ``mass_ratio`` run needs a primary mass, and there is no m1 axis to
read it off. Rather than pick a value, ``_window_factors`` integrates the window
against that parameter's own parametric model,

    log W = log int dx p(x | theta) w(x, grid) - log int dx p(x | theta),

which is a log probability: an open window gives exactly 0 whatever the model,
and a window admitting half of p's support gives log(1/2). Dropping the second
term would leave p's own normalization in the published rate.

The models here are toys, so the arithmetic is checkable in closed form rather
than against another implementation of the same thing.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp as LSE

from pixelpop.result.save_popsummary import (
    WINDOW_MARGINAL_POINTS,
    _window_factors,
    _window_marginal_parameters,
)
from pixelpop.utils.data import PixelPopData

BINS = [6, 4]
Q_CUT = 0.5
M_CUT = 3.0


# --- toy models ------------------------------------------------------------
# Each indexes `data` directly, so a missing input raises KeyError -- which is
# how _window_marginal_parameters discovers what a window needs.

def open_window(data, unused):
    """Admits everything: log w = 0, but still reads a marginalized parameter."""
    return jnp.zeros_like(data['log_mass_1'] + data['mass_ratio'])


def q_window(data, q_cut):
    """Depends only on the pixelated axis, so nothing has to be marginalized."""
    return jnp.where(data['mass_ratio'] > q_cut, 0., -jnp.inf)


def m1_window(data, m_cut):
    """A hard cut on the *non*-pixelated primary mass: pure marginalization."""
    return jnp.where(data['log_mass_1'] > m_cut, 0., -jnp.inf) \
        + jnp.zeros_like(data['mass_ratio'])


def m2_window(data, m_cut):
    """The real shape: a secondary-mass cut, m2 = m1 * q, mixing both."""
    return jnp.where(data['log_mass_1'] + jnp.log(data['mass_ratio']) > m_cut,
                     0., -jnp.inf)


def flat_m1(data, slope):
    """Unnormalized log p(m1); the factor divides its normalization out."""
    return slope * data['log_mass_1']


def conditional_m1(data, slope):
    """log p(m1 | q): a *conditional* model, which only has an answer once the
    grid supplies q. This is the shape of the real p(q | m1)."""
    return slope * data['log_mass_1'] * data['mass_ratio']


def _data(pixelpop_parameters, other_parameters, models, hypers, priors):
    rng = np.random.default_rng(0)
    nobs, npe, ninj = 3, 40, 500
    def block(shape):
        return {
            'log_mass_1': jnp.asarray(rng.uniform(np.log(3), np.log(100), shape)),
            'mass_ratio': jnp.asarray(rng.uniform(0.05, 1., shape)),
            'redshift': jnp.asarray(rng.uniform(0.05, 1.2, shape)),
            'log_prior': jnp.zeros(shape),
            }
    posteriors, injections = block((nobs, npe)), block(ninj)
    injections['total_generated'] = 1e5
    injections['analysis_time'] = 1.0
    return PixelPopData(
        name='window_test',
        posteriors=posteriors, injections=injections,
        pixelpop_parameters=list(pixelpop_parameters),
        other_parameters=list(other_parameters),
        bins=BINS[:len(pixelpop_parameters)],
        minima={'log_mass_1': float(np.log(3)), 'mass_ratio': 0.05},
        maxima={'log_mass_1': float(np.log(100)), 'mass_ratio': 1.0},
        parametric_models=dict(models),
        parameter_to_hyperparameters=dict(hypers),
        priors=dict(priors),
        )


@pytest.fixture
def q_grid_m1_marginal():
    """mass_ratio pixelated, m1 parametric: the advertised marginalization case."""
    return _data(
        ['mass_ratio'],
        ['log_mass_1', 'mass_ratio_window'],
        {'log_mass_1': flat_m1, 'mass_ratio_window': m2_window},
        {'log_mass_1': ['slope'], 'mass_ratio_window': ['m_cut']},
        {'slope': ([0., 1.], __import__('numpyro').distributions.Uniform),
         'm_cut': ([0., 5.], __import__('numpyro').distributions.Uniform)},
        )


def _centres(pixelpop_data, axis=0):
    edges = np.asarray(pixelpop_data.bin_axes[axis])
    return 0.5 * (edges[:-1] + edges[1:])


def _marginal_grid(pixelpop_data, par):
    return np.linspace(pixelpop_data.minima[par], pixelpop_data.maxima[par],
                       WINDOW_MARGINAL_POINTS)


# --- no marginalization ----------------------------------------------------

def test_a_window_on_the_grid_alone_is_evaluated_at_the_bin_centres():
    """Nothing to integrate over: the factor is just w(centres), so a bin the
    window closes goes to -inf and one it leaves open contributes nothing."""
    pp = _data(['mass_ratio'], ['mass_ratio_window'],
               {'mass_ratio_window': q_window},
               {'mass_ratio_window': ['q_cut']},
               {'q_cut': ([0., 1.], __import__('numpyro').distributions.Uniform)})
    hyperposterior = {'q_cut': np.array([Q_CUT]),
                      'log_marginal_mass_ratio': np.zeros((1, BINS[0]))}

    factors = _window_factors(hyperposterior, pp, 1)
    centres = _centres(pp)

    assert factors.shape == (1, BINS[0])
    expected = np.where(centres > Q_CUT, 0., -np.inf)
    np.testing.assert_allclose(factors[0], expected)


# --- marginalizing over one extra parameter --------------------------------

def test_an_open_window_contributes_exactly_nothing(q_grid_m1_marginal):
    """The normalization test. w = 1 everywhere, so W = 1 for *any* p(x) -- if
    the second term were dropped, this would come back as log int p dx instead,
    i.e. the model's normalization smeared into the rate."""
    pp = q_grid_m1_marginal
    pp.parametric_models['mass_ratio_window'] = open_window
    pp.parameter_to_hyperparameters['mass_ratio_window'] = ['m_cut']
    hyperposterior = {'m_cut': np.array([M_CUT]), 'slope': np.array([2.5]),
                      'log_marginal_mass_ratio': np.zeros((1, BINS[0]))}

    factors = _window_factors(hyperposterior, pp, 1)
    np.testing.assert_allclose(factors[0], np.zeros(BINS[0]), atol=1e-6)


def test_a_marginalized_window_is_the_log_fraction_of_the_model_it_admits(
        q_grid_m1_marginal):
    """A cut on the marginalized parameter alone gives the same number in every
    bin: log P(m1 > cut), under p(m1) rather than under a flat average."""
    pp = q_grid_m1_marginal
    pp.parametric_models['mass_ratio_window'] = m1_window
    slope = 2.5
    hyperposterior = {'m_cut': np.array([M_CUT]), 'slope': np.array([slope]),
                      'log_marginal_mass_ratio': np.zeros((1, BINS[0]))}

    factors = _window_factors(hyperposterior, pp, 1)

    x = _marginal_grid(pp, 'log_mass_1')
    log_p = slope * x
    expected = float(LSE(jnp.asarray(log_p[x > M_CUT]))
                     - LSE(jnp.asarray(log_p)))

    np.testing.assert_allclose(factors[0], np.full(BINS[0], expected), rtol=1e-5)
    assert -np.inf < expected < 0.        # a real fraction, admitted in part


def test_the_weighting_follows_the_model_not_the_grid(q_grid_m1_marginal):
    """Same window, different p(m1): a model pushed above the cut has more of its
    support admitted. Catches an implementation that averages the window flatly
    over the marginal grid and ignores p."""
    pp = q_grid_m1_marginal
    pp.parametric_models['mass_ratio_window'] = m1_window
    base = {'m_cut': np.array([M_CUT]),
            'log_marginal_mass_ratio': np.zeros((1, BINS[0]))}

    steep = _window_factors(dict(base, slope=np.array([6.])), pp, 1)[0, 0]
    shallow = _window_factors(dict(base, slope=np.array([0.5])), pp, 1)[0, 0]

    assert steep > shallow      # more mass above the cut

    x = _marginal_grid(pp, 'log_mass_1')
    flat = np.log(np.mean(x > M_CUT))
    assert not np.isclose(steep, flat, rtol=1e-3)


def test_a_window_mixing_grid_and_marginal_axes_varies_across_the_grid(
        q_grid_m1_marginal):
    """m2 = m1*q: the admitted fraction of p(m1) depends on which q bin it is,
    so the factor must vary along the grid and increase with q."""
    pp = q_grid_m1_marginal
    slope = 2.0
    hyperposterior = {'m_cut': np.array([M_CUT]), 'slope': np.array([slope]),
                      'log_marginal_mass_ratio': np.zeros((1, BINS[0]))}

    factors = _window_factors(hyperposterior, pp, 1)[0]
    assert (np.diff(factors) > 0).all()

    x = _marginal_grid(pp, 'log_mass_1')
    log_p = slope * x
    for jj, q in enumerate(_centres(pp)):
        admitted = (x + np.log(q)) > M_CUT
        expected = float(LSE(jnp.asarray(log_p[admitted])) - LSE(jnp.asarray(log_p)))
        np.testing.assert_allclose(factors[jj], expected, rtol=1e-5)


def test_a_conditional_model_is_integrated_at_each_grid_point(q_grid_m1_marginal):
    """p(m1 | q) has no value until the grid supplies q. The model is handed the
    grid alongside its own axis, so each bin integrates against its own density;
    evaluating it on the bare marginal grid instead raises KeyError('mass_ratio')
    -- which is how a real p(q | m1) used to fail here."""
    pp = q_grid_m1_marginal
    pp.parametric_models['log_mass_1'] = conditional_m1
    slope = 3.0
    hyperposterior = {'m_cut': np.array([M_CUT]), 'slope': np.array([slope]),
                      'log_marginal_mass_ratio': np.zeros((1, BINS[0]))}

    factors = _window_factors(hyperposterior, pp, 1)[0]

    x = _marginal_grid(pp, 'log_mass_1')
    for jj, q in enumerate(_centres(pp)):
        log_p = slope * x * q                      # the density in *this* bin
        admitted = (x + np.log(q)) > M_CUT
        expected = float(LSE(jnp.asarray(log_p[admitted])) - LSE(jnp.asarray(log_p)))
        np.testing.assert_allclose(factors[jj], expected, rtol=1e-5)

    # and it is genuinely conditional: a flat p(m1) gives different numbers
    pp.parametric_models['log_mass_1'] = flat_m1
    flat = _window_factors(hyperposterior, pp, 1)[0]
    assert not np.allclose(flat, factors)


def test_windows_accumulate(q_grid_m1_marginal):
    """Two windows on the same parameter add in the log, so the rate is
    multiplied by both."""
    pp = q_grid_m1_marginal
    hyperposterior = {'m_cut': np.array([M_CUT]), 'slope': np.array([2.0]),
                      'log_marginal_mass_ratio': np.zeros((1, BINS[0]))}
    one = _window_factors(hyperposterior, pp, 1)

    pp.window_parameters = ['mass_ratio', 'mass_ratio']
    two = _window_factors(hyperposterior, pp, 1)
    np.testing.assert_allclose(two, 2 * one)


def test_every_hyperposterior_sample_gets_its_own_factor(q_grid_m1_marginal):
    """The factor is a function of the hyperparameters, so it cannot be computed
    once and reused across the chain."""
    pp = q_grid_m1_marginal
    hyperposterior = {'m_cut': np.array([2.0, 4.0]), 'slope': np.array([2.0, 2.0]),
                      'log_marginal_mass_ratio': np.zeros((2, BINS[0]))}

    factors = _window_factors(hyperposterior, pp, 2)
    assert factors.shape == (2, BINS[0])
    assert (factors[0] > factors[1]).all()      # the looser cut admits more


# --- which parameters get marginalized -------------------------------------

def test_only_the_parameters_the_window_needs_are_marginalized(q_grid_m1_marginal):
    """The candidate list is everything parametric with grid bounds; integrating
    over one the window never reads would be wasted work, and would silently
    change nothing only if the normalization is right."""
    pp = q_grid_m1_marginal
    grid_data = {'mass_ratio': _centres(pp)}
    candidates = ['log_mass_1', 'redshift']
    grids = {p: np.linspace(0., 1., 8) for p in candidates}

    assert _window_marginal_parameters(m2_window, [M_CUT], grid_data,
                                       candidates, grids) == ['log_mass_1']
    assert _window_marginal_parameters(q_window, [Q_CUT], grid_data,
                                       candidates, grids) == []


def test_a_window_needing_something_unavailable_reraises(q_grid_m1_marginal):
    """With no candidate able to supply the missing input, the KeyError from the
    model itself is the useful error -- not an empty list and a later crash."""
    pp = q_grid_m1_marginal
    grid_data = {'mass_ratio': _centres(pp)}
    with pytest.raises(KeyError):
        _window_marginal_parameters(m2_window, [M_CUT], grid_data, [], {})
