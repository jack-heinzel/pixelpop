"""
Tests for the worst-R-hat / worst-Neff diagnostic search.

Mostly: slicing the big sites and filtering to the latents must return exactly
what a bulk ``summary`` call would have.
"""
import numpy as np
import pytest
from numpyro.diagnostics import summary

from pixelpop.models.probabilistic import (
    BULK_DIAGNOSTIC_PARAMETERS,
    get_worst_rhat_neff,
)

DRAWS = 40


def reference(samples, key):
    """numpyro's own answer for one site: (max r_hat, its index), (min n_eff, its index)."""
    site = summary({key: samples[key]}, group_by_chain=False)[key]
    rhat, neff = np.asarray(site['r_hat']), np.asarray(site['n_eff'])
    return ((rhat.reshape(-1).max(), np.unravel_index(np.argmax(rhat), rhat.shape)),
            (neff.reshape(-1).min(), np.unravel_index(np.argmin(neff), neff.shape)))


@pytest.fixture
def samples():
    """A run's worth of sites: a pixel field, a vector, and some scalars."""
    rng = np.random.default_rng(0)
    out = {
        # over BULK_DIAGNOSTIC_PARAMETERS, so it takes the sliced path
        '_eigenbasis_sites': rng.standard_normal((DRAWS, 20, 20, 20)).astype(np.float32),
        'merger_rate_density': rng.standard_normal((DRAWS, 20, 20, 20)).astype(np.float32),
        'log_ranges': rng.standard_normal((DRAWS, 3)).astype(np.float32),
        'lnsigma': rng.standard_normal(DRAWS).astype(np.float32),
        'Nexp': rng.standard_normal(DRAWS).astype(np.float32),
    }
    # a drifting parameter, so the worst r_hat sits in the field
    out['_eigenbasis_sites'][:, 3, 4, 5] += np.linspace(0., 8., DRAWS)
    return out


LATENT = {'_eigenbasis_sites', 'log_ranges', 'lnsigma'}


# ---------------------------------------------------------------------------
# Agreement with numpyro
# ---------------------------------------------------------------------------

def test_sliced_site_matches_a_bulk_summary_call(samples):
    """Slicing the leading event axis changes nothing."""
    key = '_eigenbasis_sites'
    (rhat, rhat_index), (neff, neff_index) = reference(samples, key)

    rhat_key, rhat_chain, neff_key, neff_chain = get_worst_rhat_neff(
        samples, latent_sites={key}
    )
    assert rhat_key == f'{key}{[int(p) for p in rhat_index]}'
    assert neff_key == f'{key}{[int(p) for p in neff_index]}'
    np.testing.assert_array_equal(rhat_chain, samples[key][(..., *rhat_index)])
    np.testing.assert_array_equal(neff_chain, samples[key][(..., *neff_index)])


def test_planted_drift_is_found(samples):
    assert get_worst_rhat_neff(samples, latent_sites=LATENT)[0] == '_eigenbasis_sites[3, 4, 5]'


def test_scalar_site_is_labelled_without_an_index(samples):
    only_scalars = {k: samples[k] for k in ('lnsigma', 'Nexp')}
    rhat_key, rhat_chain, neff_key, _ = get_worst_rhat_neff(only_scalars)
    assert rhat_key in ('lnsigma', 'Nexp') and '[' not in rhat_key
    assert neff_key in ('lnsigma', 'Nexp') and '[' not in neff_key
    np.testing.assert_array_equal(rhat_chain, only_scalars[rhat_key])


def test_vector_site_is_labelled_with_one_index(samples):
    rhat_key, rhat_chain, _, _ = get_worst_rhat_neff(samples, latent_sites={'log_ranges'})
    assert rhat_key.startswith('log_ranges[')
    assert rhat_chain.shape == (DRAWS,)


def test_bulk_and_sliced_paths_agree_across_the_threshold():
    """A site either side of BULK_DIAGNOSTIC_PARAMETERS must give the same answer."""
    rng = np.random.default_rng(1)
    width = int(np.ceil(np.sqrt(BULK_DIAGNOSTIC_PARAMETERS))) + 1  # width^2 > threshold
    field = rng.standard_normal((DRAWS, width, width)).astype(np.float32)
    field[:, 2, 3] += np.linspace(0., 8., DRAWS)
    assert field[0].size > BULK_DIAGNOSTIC_PARAMETERS  # takes the sliced path

    (_, rhat_index), (_, neff_index) = reference({'f': field}, 'f')
    rhat_key, _, neff_key, _ = get_worst_rhat_neff({'f': field})
    assert rhat_key == f'f{[int(p) for p in rhat_index]}'
    assert neff_key == f'f{[int(p) for p in neff_index]}'


# ---------------------------------------------------------------------------
# Which sites get searched
# ---------------------------------------------------------------------------

def test_latent_sites_excludes_deterministic_sites(samples):
    """merger_rate_density is a function of the latents, so it is never reported."""
    rhat_key, _, neff_key, _ = get_worst_rhat_neff(samples, latent_sites=LATENT)
    assert not rhat_key.startswith('merger_rate_density')
    assert not neff_key.startswith('merger_rate_density')


def test_without_latent_sites_every_key_is_searched(samples):
    """Callers that pass nothing still see all sites."""
    reported = set()
    for key in ('merger_rate_density', '_eigenbasis_sites'):
        worst = reference(samples, key)
        reported.add(worst[0][0])
    rhat_key, _, _, _ = get_worst_rhat_neff(samples)
    assert rhat_key.split('[')[0] in samples


def test_skip_keys_matches_the_site_name(samples):
    """Matched on the site name, not the indexed name the old version used."""
    rhat_key, _, neff_key, _ = get_worst_rhat_neff(
        samples, skip_keys=['_eigenbasis_sites', 'merger_rate_density']
    )
    for key in (rhat_key, neff_key):
        assert not key.startswith('_eigenbasis_sites')
        assert not key.startswith('merger_rate_density')


def test_filters_that_exclude_everything_fall_back(samples):
    """A filter matching nothing must not leave the run with nothing to print."""
    rhat_key, _, _, _ = get_worst_rhat_neff(samples, latent_sites={'not_a_site'})
    assert rhat_key.split('[')[0] in samples


def test_nan_diagnostics_still_report_something():
    """A constant site gives NaN r_hat/Neff, and NaN loses every comparison. The
    search must still come back with a usable key rather than dereferencing None."""
    rng = np.random.default_rng(2)
    constant = {'stuck': np.ones(DRAWS, dtype=np.float32)}
    rhat_key, rhat_chain, neff_key, _ = get_worst_rhat_neff(constant)
    assert rhat_key == 'stuck' and neff_key == 'stuck'
    np.testing.assert_array_equal(rhat_chain, constant['stuck'])

    mixed = dict(constant, moving=rng.standard_normal(DRAWS).astype(np.float32))
    assert get_worst_rhat_neff(mixed)[0] in ('stuck', 'moving')


def test_skip_keys_and_latent_sites_compose(samples):
    rhat_key, _, neff_key, _ = get_worst_rhat_neff(
        samples, skip_keys=['_eigenbasis_sites'], latent_sites=LATENT
    )
    for key in (rhat_key, neff_key):
        assert key.split('[')[0] in {'log_ranges', 'lnsigma'}
