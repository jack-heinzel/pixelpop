"""
Tests for ``PixelPopRateFunction``, the model wrapper handed to ``population_error``.

The rate likelihood has ``Nexp = live_time * mean(injection weights)``
(``rate_likelihood``), but ``population_error``'s rate mode has no live time: it
takes the injection weights as estimating Nexp directly. Before the fix the
injection rate function returned the rate per unit time, so every selection
covariance -- and the selection term of the likelihood correction -- came out
too small by ``live_time**2`` (9.3 for the GWTC-5.0 runs, ``analysis_time`` = 3.05).
"""
import jax.numpy as jnp
import numpy as np
import population_error
import pytest

from pixelpop.models.gwpop_models import rate_likelihood
from pixelpop.result.post_processing import PixelPopRateFunction
from pixelpop.utils.data import PixelPopData

BINS = [7, 5]
LIVE_TIME = 3.0
TOTAL = 1e6


@pytest.fixture(scope='module')
def pixelpop_data():
    rng = np.random.default_rng(0)
    nobs, npe, ninj = 4, 50, 3000
    posteriors = {
        'log_mass_1': jnp.asarray(rng.uniform(np.log(6), np.log(40), (nobs, npe))),
        'redshift': jnp.asarray(rng.uniform(0.05, 1.2, (nobs, npe))),
        'log_prior': jnp.zeros((nobs, npe)),
    }
    injections = {
        'log_mass_1': jnp.asarray(rng.uniform(np.log(6), np.log(40), ninj)),
        'redshift': jnp.asarray(rng.uniform(0.05, 1.2, ninj)),
        'log_prior': jnp.zeros(ninj),
        'total_generated': TOTAL,
        'analysis_time': LIVE_TIME,
    }
    return PixelPopData(
        name='rate_function_test',
        posteriors=posteriors,
        injections=injections,
        pixelpop_parameters=['log_mass_1', 'redshift'],
        other_parameters=[],
        bins=list(BINS),
        minima={'log_mass_1': float(np.log(3)), 'redshift': 0.0},
        maxima={'log_mass_1': float(np.log(100)), 'redshift': 1.5},
        )


def _sample(seed=1):
    return {'merger_rate_density': jnp.asarray(np.random.default_rng(seed).normal(size=tuple(BINS)))}


def _injection_weights(pixelpop_data, sample):
    rate = PixelPopRateFunction(pixelpop_data, dataset_type='injections')
    return np.asarray(rate(pixelpop_data.injections, sample)) / np.exp(np.asarray(pixelpop_data.injections['log_prior']))


def test_only_injections_are_scaled_by_the_live_time(pixelpop_data):
    sample = _sample()
    for dataset_type, data, bins, scale in [
            ('posteriors', pixelpop_data.posteriors, pixelpop_data.event_bins, 1.),
            ('injections', pixelpop_data.injections, pixelpop_data.inj_bins, LIVE_TIME)]:
        rate = PixelPopRateFunction(pixelpop_data, dataset_type=dataset_type)
        expected = scale * np.exp(np.asarray(sample['merger_rate_density'])[bins] + np.asarray(data['ln_dVTc']))
        np.testing.assert_allclose(np.asarray(rate(data, sample)), expected, rtol=1e-5)


def test_injection_weights_average_to_nexp(pixelpop_data):
    sample = _sample()
    weights = _injection_weights(pixelpop_data, sample)
    event_weights = jnp.zeros(pixelpop_data.posteriors['log_prior'].shape)
    per_time = jnp.log(weights / LIVE_TIME)
    nexp = rate_likelihood(event_weights, per_time, TOTAL, live_time=LIVE_TIME)['nexp']
    np.testing.assert_allclose(weights.sum() / TOTAL, float(nexp), rtol=1e-5)


def test_selection_variance_seen_by_population_error_matches_the_likelihood(pixelpop_data):
    # population_error's selection covariance features are the injection weights scaled by
    # 1/sqrt(N (N - 1)); their norm is its Var(Nexp). It must equal the selection variance of
    # the rate likelihood, up to the O(Nexp^2 / N) term it drops.
    sample = _sample()
    weights = _injection_weights(pixelpop_data, sample)
    seen = np.sum(weights**2) / (TOTAL * (TOTAL - 1))
    event_weights = jnp.zeros(pixelpop_data.posteriors['log_prior'].shape)
    truth = float(rate_likelihood(event_weights, jnp.log(weights / LIVE_TIME), TOTAL, live_time=LIVE_TIME)['total_vt_lnL_variance'])
    np.testing.assert_allclose(seen, truth, rtol=1e-2)


def test_selection_precision_scales_with_the_live_time_squared(pixelpop_data):
    # end to end through population_error: the selection precision statistic is proportional to
    # live_time^2 and the single-event precision does not depend on it
    rng = np.random.default_rng(2)
    nsamples = 40
    hyperposterior = {'merger_rate_density': jnp.asarray(rng.normal(scale=0.3, size=(nsamples,) + tuple(BINS)))}
    posteriors = dict(pixelpop_data.posteriors, prior=jnp.exp(pixelpop_data.posteriors['log_prior']))
    stats = {}
    for live_time in [1.0, LIVE_TIME]:
        injections = dict(pixelpop_data.injections, analysis_time=live_time)
        injections['prior'] = jnp.exp(injections['log_prior'])
        data = PixelPopData.__new__(PixelPopData)
        data.__dict__.update(pixelpop_data.__dict__)
        data.injections = injections
        stats[live_time] = population_error.error_statistics(
            PixelPopRateFunction(data, dataset_type='posteriors'), injections, posteriors, dict(hyperposterior),
            vt_model_function=PixelPopRateFunction(data, dataset_type='injections'),
            include_likelihood_correction=True, rate=True, verbose=False)
    np.testing.assert_allclose(stats[LIVE_TIME]['selection_precision_statistic'],
                               LIVE_TIME**2 * stats[1.0]['selection_precision_statistic'], rtol=1e-6)
    np.testing.assert_allclose(stats[LIVE_TIME]['event_precision_statistic'], stats[1.0]['event_precision_statistic'], rtol=1e-10)
