import unittest
import numpy as np
import jax.numpy as jnp
import pixelpop
from pixelpop.utils.data import (
    convert_m1q_to_lm1m2,
    convert_m1_to_lm1,
    convert_m1m2_to_lm1lm2,
    clean_par,
    check_bins,
    posteriors_to_rectangular,
    PixelPopData,
)

class TestConvertMasses(unittest.TestCase):

    def test_convert_m1q_to_lm1m2(self):
        data = {
            "mass_1": jnp.array([10.0]),
            "mass_ratio": jnp.array([0.5]),
            "prior": jnp.array([1.0]),
        }

        out = convert_m1q_to_lm1m2(data)

        self.assertTrue("log_mass_1" in out)
        self.assertTrue("log_mass_2" in out)
        self.assertTrue("log_prior" in out)

        self.assertAlmostEqual(
            float(out["log_mass_1"][0]),
            jnp.log(10.0),
        )
        self.assertAlmostEqual(
            float(out["log_mass_2"][0]),
            jnp.log(10.0 * 0.5),
        )
        self.assertAlmostEqual(
            float(out["log_prior"][0]),
            jnp.log(1.) + jnp.log(10.0 * 0.5),
        )

    def test_convert_m1_to_lm1(self):
        data = {
            "mass_1": jnp.array([20.0]),
            "prior": jnp.array([2.0]),
        }

        out = convert_m1_to_lm1(data)

        self.assertAlmostEqual(
            float(out["log_mass_1"][0]),
            jnp.log(20.0),
        )
        self.assertAlmostEqual(
            float(out["log_prior"][0]),
            jnp.log(2.0) + jnp.log(20.0),
        )

    def test_convert_m1m2_to_lm1lm2(self):
        data = {
            "mass_1": jnp.array([30.0]),
            "mass_2": jnp.array([10.0]),
            "prior": jnp.array([1.0]),
        }

        out = convert_m1m2_to_lm1lm2(data)

        self.assertAlmostEqual(float(out["log_mass_1"][0]), jnp.log(30.0))
        self.assertAlmostEqual(float(out["log_mass_2"][0]), jnp.log(10.0))
        self.assertAlmostEqual(
            float(out["log_prior"][0]),
            jnp.log(30.0) + jnp.log(10.0),
        )


class TestCleanPar(unittest.TestCase):

    def test_clean_par_replacement(self):
        data = {
            "x": jnp.array([0.0, 5.0, 20.0]),
            "log_prior": jnp.zeros(3),
        }

        out = clean_par(data, "x", minimum=1.0, maximum=10.0)

        self.assertTrue(jnp.isinf(out["log_prior"][0]))
        self.assertTrue(jnp.isinf(out["log_prior"][2]))
        self.assertFalse(jnp.isinf(out["log_prior"][1]))

    def test_clean_par_removal(self):
        data = {
            "x": jnp.array([0.0, 5.0, 20.0]),
            "y": jnp.array([1.0, 2.0, 3.0]),
        }

        out = clean_par(data, "x", 1.0, 10.0, remove=True)

        self.assertEqual(len(out["x"]), 1)
        self.assertEqual(float(out["x"][0]), 5.0)
        self.assertEqual(float(out["y"][0]), 2.0)


class TestCheckBins(unittest.TestCase):

    def test_check_bins_success(self):
        event_bins = (jnp.array([[0, 1, 2, 3, 4]]),)
        inj_bins = (jnp.array([0, 1, 2, 3, 4]),)

        success, e_bad, i_bad = check_bins(event_bins, inj_bins, bins=5)

        self.assertTrue(success)
        self.assertTrue(jnp.all(e_bad == 0))
        self.assertTrue(jnp.all(i_bad == 0))

    def test_check_bins_injection_free(self):
        event_bins = (jnp.array([[0, 1, 2, 3, 4]]),)
        inj_bins = (jnp.array([0, 1, 3, 4]),)

        success, e_bad, _ = check_bins(event_bins, inj_bins, bins=5)

        self.assertFalse(success)
        self.assertTrue(jnp.isinf(e_bad[0,2]))


class TestPosteriorsToRectangular(unittest.TestCase):
    """posteriors_to_rectangular pads short events to a common width with prior=inf."""

    def _events(self, counts=(50, 120, 300)):
        rng = np.random.default_rng(0)
        return {
            f'event_{ii}': {
                'log_mass_1': rng.uniform(np.log(6), np.log(40), n),
                'redshift': rng.uniform(0.05, 1.2, n),
                'prior': rng.uniform(0.1, 1.0, n),
            }
            for ii, n in enumerate(counts)
        }

    def test_padding_shapes_counts_and_inf(self):
        counts = (50, 120, 300)
        npe = max(counts)
        rect, event_counts, names = posteriors_to_rectangular(
            self._events(counts), ['log_mass_1', 'redshift'], npe
        )

        self.assertEqual(len(names), len(counts))
        # rectangular: every key is (Nobs, npe), 'prior' included
        for key in ('log_mass_1', 'redshift', 'prior'):
            self.assertEqual(rect[key].shape, (len(counts), npe))
        # real (un-padded) counts are recovered
        np.testing.assert_array_equal(np.asarray(event_counts), np.asarray(counts))

        for ii, n in enumerate(counts):
            prior_row = np.asarray(rect['prior'][ii])
            # the first n entries are real (finite); the padded tail is +inf
            self.assertTrue(np.all(np.isfinite(prior_row[:n])))
            self.assertTrue(np.all(np.isinf(prior_row[n:])))

    def test_downsampling_truncates(self):
        counts = (50, 120, 300)
        npe = 80  # below the largest event -> that event is downsampled
        rect, event_counts, _ = posteriors_to_rectangular(
            self._events(counts), ['log_mass_1', 'redshift'], npe
        )
        self.assertEqual(rect['prior'].shape, (len(counts), npe))
        np.testing.assert_array_equal(
            np.asarray(event_counts), np.minimum(np.asarray(counts), npe)
        )


class TestPaddedPixelPopData(unittest.TestCase):
    """PixelPopData on a padded rectangular set tracks real per-event counts."""

    def _build(self, counts=(50, 120, 300)):
        rng = np.random.default_rng(1)
        npe = max(counts)
        events = {
            f'event_{ii}': {
                'log_mass_1': rng.uniform(np.log(6), np.log(40), n),
                'redshift': rng.uniform(0.05, 1.2, n),
                'prior': rng.uniform(0.1, 1.0, n),
            }
            for ii, n in enumerate(counts)
        }
        rect, event_counts, _ = posteriors_to_rectangular(
            events, ['log_mass_1', 'redshift'], npe
        )
        # prior=inf on padded rows -> log_prior=inf (zero weight)
        rect['log_prior'] = jnp.log(rect.pop('prior'))

        n_inj = 4000
        injections = {
            'log_mass_1': jnp.asarray(rng.uniform(np.log(6), np.log(40), n_inj)),
            'redshift': jnp.asarray(rng.uniform(0.05, 1.2, n_inj)),
            'log_prior': jnp.zeros(n_inj),
            'total_generated': 1e6,
            'analysis_time': 1.0,
        }
        pp = PixelPopData(
            name='padded_test',
            posteriors=rect,
            injections=injections,
            pixelpop_parameters=['log_mass_1', 'redshift'],
            other_parameters=[],
            bins=10,
            minima={'log_mass_1': float(np.log(3)), 'redshift': 0.0},
            maxima={'log_mass_1': float(np.log(100)), 'redshift': 2.0},
            event_counts=event_counts,
        )
        return pp, npe

    def test_padded_construction(self):
        counts = (50, 120, 300)
        npe = max(counts)
        pp, npe = self._build(counts)

        self.assertEqual(pp.Nobs, len(counts))
        # rectangular bins: one index array per pixelpop dimension, each (Nobs, npe)
        self.assertEqual(len(pp.event_bins), pp.dimension)
        self.assertEqual(pp.event_bins[0].shape, (len(counts), npe))
        self.assertEqual(pp.posteriors['log_prior'].shape, (len(counts), npe))
        # real per-event counts are preserved on the object
        np.testing.assert_array_equal(np.asarray(pp.event_counts), np.asarray(counts))

    def test_default_event_counts_is_npe(self):
        # when event_counts is not supplied, it defaults to NPE for every event
        rng = np.random.default_rng(2)
        npe = 200
        posteriors = {
            'log_mass_1': jnp.asarray(rng.uniform(np.log(6), np.log(40), (3, npe))),
            'redshift': jnp.asarray(rng.uniform(0.05, 1.2, (3, npe))),
            'log_prior': jnp.zeros((3, npe)),
        }
        injections = {
            'log_mass_1': jnp.asarray(rng.uniform(np.log(6), np.log(40), 4000)),
            'redshift': jnp.asarray(rng.uniform(0.05, 1.2, 4000)),
            'log_prior': jnp.zeros(4000),
            'total_generated': 1e6,
            'analysis_time': 1.0,
        }
        pp = PixelPopData(
            name='default_counts',
            posteriors=posteriors,
            injections=injections,
            pixelpop_parameters=['log_mass_1', 'redshift'],
            other_parameters=[],
            bins=10,
            minima={'log_mass_1': float(np.log(3)), 'redshift': 0.0},
            maxima={'log_mass_1': float(np.log(100)), 'redshift': 2.0},
        )
        np.testing.assert_array_equal(np.asarray(pp.event_counts), np.full(3, npe))
