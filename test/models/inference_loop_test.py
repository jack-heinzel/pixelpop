"""
Tests for inference_loop's draw-collection loop.

The chain is preallocated at ``tot_samples`` and filled a chunk at a time, so
these check the collected draws are what the chunks actually produced and that
the unwritten tail never escapes -- into the returned dict, the saved h5, or the
diagnostics.
"""
import h5py
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random

from pixelpop.models.probabilistic import inference_loop

TOT_SAMPLES = 12
NUM_SAMPLES = 4  # -> 3 chunks


def toy_model():
    numpyro.sample('x', dist.Normal(0., 1.))
    numpyro.sample('field', dist.Normal(0., 1.), sample_shape=(2, 3))


@pytest.fixture
def run(tmp_path):
    samples, _ = inference_loop(
        toy_model, warmup=50, tot_samples=TOT_SAMPLES, thinning=1,
        num_samples=NUM_SAMPLES, parallel=1, run_dir=str(tmp_path), name='toy',
        print_keys=['x'], rng_key=random.PRNGKey(0),
    )
    return samples, tmp_path / 'toy' / 'chain_0_samples.h5'


def test_returns_exactly_tot_samples(run):
    samples, _ = run
    assert len(samples) == 1
    for value in samples[0].values():
        assert value.shape[0] == TOT_SAMPLES


def test_shapes_keep_the_site_event_shape(run):
    samples, _ = run
    assert samples[0]['x'].shape == (TOT_SAMPLES,)
    assert samples[0]['field'].shape == (TOT_SAMPLES, 2, 3)


def test_no_unwritten_tail_in_the_returned_chain(run):
    """A preallocated buffer that is under-filled would leave uninitialised
    values at the end; every draw here must be a real one."""
    samples, _ = run
    for key, value in samples[0].items():
        assert np.all(np.isfinite(value)), key
        # np.empty tails show up as exact repeats or zeros; real draws do neither
        assert not np.all(value[-1] == 0), key


def test_saved_h5_matches_the_returned_samples(run):
    samples, path = run
    with h5py.File(path, 'r') as f:
        for key, value in samples[0].items():
            np.testing.assert_array_equal(f[key][()], value)


def test_saved_h5_is_not_padded_to_the_buffer(run):
    """The h5 is written from a view of the filled part, not the whole buffer."""
    _, path = run
    with h5py.File(path, 'r') as f:
        for key in f:
            assert f[key].shape[0] == TOT_SAMPLES


def test_draws_differ_across_chunks(run):
    """Guards against a slice-assignment bug that rewrites the same chunk: draw
    NUM_SAMPLES apart come from different mcmc.run calls and must not match."""
    samples, _ = run
    x = samples[0]['x']
    assert not np.allclose(x[:NUM_SAMPLES], x[NUM_SAMPLES:2 * NUM_SAMPLES])


def test_last_chunk_is_saved_when_cache_cadence_does_not_divide(tmp_path):
    """With 3 chunks and cache_cadence=2 the block fires on 0 and 2; the guard is
    that chunk 2 is the last one, so the h5 must still hold every draw."""
    samples, _ = inference_loop(
        toy_model, warmup=50, tot_samples=TOT_SAMPLES, thinning=1,
        num_samples=NUM_SAMPLES, parallel=1, run_dir=str(tmp_path), name='cadence',
        print_keys=['x'], rng_key=random.PRNGKey(0), cache_cadence=2,
    )
    with h5py.File(tmp_path / 'cadence' / 'chain_0_samples.h5', 'r') as f:
        assert f['x'].shape[0] == TOT_SAMPLES
        np.testing.assert_array_equal(f['x'][()], samples[0]['x'])


def test_last_chunk_saved_for_any_cadence(tmp_path):
    """cache_cadence=5 against 3 chunks fires only on chunk 0 without the guard."""
    samples, _ = inference_loop(
        toy_model, warmup=50, tot_samples=TOT_SAMPLES, thinning=1,
        num_samples=NUM_SAMPLES, parallel=1, run_dir=str(tmp_path), name='wide',
        print_keys=['x'], rng_key=random.PRNGKey(0), cache_cadence=5,
    )
    with h5py.File(tmp_path / 'wide' / 'chain_0_samples.h5', 'r') as f:
        np.testing.assert_array_equal(f['x'][()], samples[0]['x'])


def test_uneven_total_collects_whole_chunks_only(tmp_path):
    """tot_samples/num_samples is truncated, so 10/4 gives 2 chunks of 4."""
    samples, _ = inference_loop(
        toy_model, warmup=50, tot_samples=10, thinning=1, num_samples=4,
        parallel=1, run_dir=str(tmp_path), name='uneven', print_keys=['x'],
        rng_key=random.PRNGKey(0),
    )
    assert samples[0]['x'].shape[0] == 8
    assert np.all(np.isfinite(samples[0]['field']))
