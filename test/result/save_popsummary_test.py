"""
Tests for the popsummary hyperparameter-list reconciliation, and for the
probability-density grids.

``popsummary.PopulationResult`` writes the ``hyperparameters`` attribute only
when it *creates* the file. Handed an existing one it opens it and keeps whatever
is stored, even though ``hyperparameters=`` was passed, and
``set_hyperparameter_samples`` then length-checks the sample array against that
stale list and nothing more. So a changed list either raises a bare axis-length
mismatch that names neither the file nor the keys, or -- if the count happens to
match -- silently mislabels every column. The second is the one worth a test:
its pre-fix behaviour is a valid-looking file with the wrong names on the data.
"""
import jax.numpy as jnp
import numpy as np
import popsummary
import pytest

from pixelpop.models.gwpop_models import COSMO
from pixelpop.result.save_popsummary import (
    _probability_density_grids,
    _reconcile_hyperparameters,
    _redshift_volume_weight,
)
from pixelpop.utils.data import PixelPopData, log_dVTc

KEYS = ['alpha', 'beta', 'lamb', 'mmax']
BINS = [7, 5]
Z_MAX = 1.5


def _result(tmp_path, hyperparameters, name='run'):
    return popsummary.popresult.PopulationResult(
        str(tmp_path / f'{name}_popsummary.h5'),
        hyperparameters=list(hyperparameters),
        verbose=False,
        )


def _stored(result):
    return [s.decode() if isinstance(s, bytes) else str(s)
            for s in result.get_metadata('hyperparameters')]


def test_an_unchanged_list_is_left_alone(tmp_path):
    result = _result(tmp_path, KEYS)
    _reconcile_hyperparameters(result, KEYS, result.fname, overwrite=True)
    assert _stored(result) == KEYS


def test_a_reordered_list_is_rewritten(tmp_path):
    """The silent case: same names, same count, different order. Nothing
    downstream length-checks its way out of this one."""
    result = _result(tmp_path, KEYS)
    reordered = [KEYS[1], KEYS[0]] + KEYS[2:]
    _reconcile_hyperparameters(result, reordered, result.fname, overwrite=True)
    assert _stored(result) == reordered


def test_added_and_removed_names_are_rewritten(tmp_path):
    result = _result(tmp_path, KEYS)
    new_keys = KEYS[:-1] + ['max_z', 'amax']
    _reconcile_hyperparameters(result, new_keys, result.fname, overwrite=True)
    assert _stored(result) == new_keys


def test_a_changed_list_raises_without_overwrite(tmp_path):
    """Without overwrite the caller has not asked for anything in the file to be
    replaced, so say what changed rather than writing over it."""
    result = _result(tmp_path, KEYS)
    with pytest.raises(ValueError, match='hyperparameter list has changed'):
        _reconcile_hyperparameters(result, KEYS[:-1], result.fname, overwrite=False)
    assert _stored(result) == KEYS


def test_reordering_raises_without_overwrite(tmp_path):
    result = _result(tmp_path, KEYS)
    reordered = [KEYS[1], KEYS[0]] + KEYS[2:]
    with pytest.raises(ValueError, match='mislabel'):
        _reconcile_hyperparameters(result, reordered, result.fname, overwrite=False)


def test_linked_metadata_blocks_the_rewrite(tmp_path):
    """Descriptions, units and latex labels are indexed positionally against the
    hyperparameter list, so rewriting it under them would leave the file
    internally inconsistent -- and popsummary enforces only their length."""
    result = _result(tmp_path, KEYS)
    result.set_metadata('hyperparameter_units', ['' for _ in KEYS], overwrite=True)
    with pytest.raises(ValueError, match='hyperparameter_units'):
        _reconcile_hyperparameters(result, KEYS[:-1], result.fname, overwrite=True)
    assert _stored(result) == KEYS


def test_a_rewritten_list_matches_the_samples_written_after_it(tmp_path):
    """The end-to-end point: after reconciliation the sample array that
    create_popsummary builds from the current keys is accepted, and the names it
    is stored against are the current ones."""
    result = _result(tmp_path, KEYS)
    new_keys = KEYS + ['log_rate']
    samples = {k: np.arange(5, dtype=float) + ii for ii, k in enumerate(new_keys)}

    with pytest.raises(Exception):
        # what every rerun hit: 5 columns against a stored list of 4
        result.set_hyperparameter_samples(
            np.array([samples[h] for h in new_keys]).T, overwrite=True)

    _reconcile_hyperparameters(result, new_keys, result.fname, overwrite=True)
    result.set_hyperparameter_samples(
        np.array([samples[h] for h in new_keys]).T, overwrite=True)

    assert _stored(result) == new_keys
    for ii, key in enumerate(new_keys):
        np.testing.assert_allclose(
            result.get_hyperparameter_samples()[:, ii], samples[key])


# ---------------------------------------------------------------------------
# Probability-density grids
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pixelpop_data():
    """A two-axis run with redshift pixelated -- the case the volume weight is
    for, and the one whose non-redshift marginals rate mode refuses to write."""
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
        'total_generated': 1e5,
        'analysis_time': 1.0,
    }
    return PixelPopData(
        name='density_test',
        posteriors=posteriors,
        injections=injections,
        pixelpop_parameters=['log_mass_1', 'redshift'],
        other_parameters=[],
        bins=list(BINS),
        minima={'log_mass_1': float(np.log(3)), 'redshift': 0.0},
        maxima={'log_mass_1': float(np.log(100)), 'redshift': Z_MAX},
        )


@pytest.fixture(scope='module')
def field():
    """A few samples of a structured log field."""
    rng = np.random.default_rng(1)
    return rng.normal(size=(3,) + tuple(BINS))


def _bin_widths(pixelpop_data):
    return np.exp(np.asarray(pixelpop_data.logdV))


def test_log_dVTc_matches_the_factor_the_likelihood_used(pixelpop_data):
    """The grid weight has to be the same quantity preprocess_cosmology folded
    into the per-sample weights, or the reported density is not the density of
    the model that was fitted."""
    z = np.asarray(pixelpop_data.injections['redshift'])
    np.testing.assert_allclose(
        log_dVTc(COSMO, z),
        np.asarray(pixelpop_data.injections['ln_dVTc']),
        rtol=1e-5)


def test_volume_weight_is_the_bin_average_not_the_centre(pixelpop_data):
    """The field is piecewise constant on a bin, so the exact weight is the
    average of dVc/dz * 1/(1+z) across it. dVc/dz curves hard over the lowest
    bins, where the midpoint value is off by ~16% on a 5-bin axis -- not
    rounding, and not in a fixed direction either, since the factor is convex at
    low z and turns over further out."""
    weight = np.exp(np.squeeze(_redshift_volume_weight(pixelpop_data)))
    edges = np.linspace(0., Z_MAX, BINS[1] + 1)

    fine = [np.trapezoid(np.exp(log_dVTc(COSMO, np.linspace(lo, hi, 2001))),
                         np.linspace(lo, hi, 2001)) / (hi - lo)
            for lo, hi in zip(edges[:-1], edges[1:])]
    np.testing.assert_allclose(weight, fine, rtol=1e-4)

    centres = np.exp(log_dVTc(COSMO, 0.5 * (edges[:-1] + edges[1:])))
    assert np.max(np.abs(weight / centres - 1.)) > 0.1
    assert weight[0] > centres[0]      # convex where dVc/dz is still climbing


def test_volume_weight_broadcasts_on_the_redshift_axis(pixelpop_data):
    """Shaped for the grid axes of a (Nsamples, *bins) field, varying on redshift
    only -- if it landed on the mass axis the weighting would be silently wrong
    rather than a shape error."""
    weight = _redshift_volume_weight(pixelpop_data)
    assert weight.shape == (1, BINS[1])
    assert np.ndim(np.squeeze(weight)) == 1


def test_grids_are_normalized(pixelpop_data, field):
    """'Probability density' means exactly this: each grid integrates to 1 with
    respect to its own bin widths."""
    weight = _redshift_volume_weight(pixelpop_data)
    log_joint, log_marginals = _probability_density_grids(field, pixelpop_data, weight)
    widths = _bin_widths(pixelpop_data)

    np.testing.assert_allclose(
        np.exp(log_joint).sum(axis=(1, 2)) * np.prod(widths), 1., rtol=1e-5)
    for ii, par in enumerate(pixelpop_data.pixelpop_parameters):
        np.testing.assert_allclose(
            np.exp(log_marginals[par]).sum(axis=1) * widths[ii], 1., rtol=1e-5)


def test_marginals_agree_with_the_joint(pixelpop_data, field):
    """They are published as separate grids, so they have to be marginals of the
    same weighted, normalized field rather than of two different ones."""
    weight = _redshift_volume_weight(pixelpop_data)
    log_joint, log_marginals = _probability_density_grids(field, pixelpop_data, weight)
    widths = _bin_widths(pixelpop_data)

    np.testing.assert_allclose(
        np.exp(log_joint).sum(axis=1) * widths[0],
        np.exp(log_marginals['redshift']), rtol=1e-5)
    np.testing.assert_allclose(
        np.exp(log_joint).sum(axis=2) * widths[1],
        np.exp(log_marginals['log_mass_1']), rtol=1e-5)


def test_the_weight_tilts_redshift_towards_the_far_bins(pixelpop_data):
    """A flat field is a constant comoving merger rate density, whose observed
    redshift distribution is dVc/dz * 1/(1+z) -- the whole point of the factor.
    Without it the same field would come back uniform in z."""
    flat = np.zeros((1,) + tuple(BINS))
    weight = _redshift_volume_weight(pixelpop_data)
    _, weighted = _probability_density_grids(flat, pixelpop_data, weight)
    _, unweighted = _probability_density_grids(flat, pixelpop_data)

    widths = _bin_widths(pixelpop_data)
    np.testing.assert_allclose(
        np.exp(unweighted['redshift'])[0], 1. / (widths[1] * BINS[1]), rtol=1e-5)

    p_z = np.exp(weighted['redshift'])[0]
    assert (np.diff(p_z) > 0).all()
    expected = np.exp(np.squeeze(weight))
    np.testing.assert_allclose(
        p_z / p_z.sum(), expected / expected.sum(), rtol=1e-5)

    # the mass axis is untouched by a weight that varies only along redshift
    np.testing.assert_allclose(
        np.exp(weighted['log_mass_1']), np.exp(unweighted['log_mass_1']), rtol=1e-5)


def test_a_masked_field_normalizes_over_its_own_support(pixelpop_data):
    """A lower-triangular run reaches this with -inf above the diagonal. The
    normalization comes from the same array as the marginals, so it covers the
    masked support rather than the full grid halved."""
    rng = np.random.default_rng(2)
    masked = rng.normal(size=(2,) + tuple(BINS))
    masked[:, :, 3:] = -np.inf

    log_joint, log_marginals = _probability_density_grids(masked, pixelpop_data)
    widths = _bin_widths(pixelpop_data)

    np.testing.assert_allclose(
        np.exp(log_joint).sum(axis=(1, 2)) * np.prod(widths), 1., rtol=1e-5)
    assert (np.exp(log_marginals['redshift'])[:, 3:] == 0.).all()
    np.testing.assert_allclose(
        np.exp(log_marginals['redshift']).sum(axis=1) * widths[1], 1., rtol=1e-5)
