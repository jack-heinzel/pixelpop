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


class TestRaggedPixelPopData(unittest.TestCase):
    """PixelPopData should accept a list of per-event dicts with different sample counts."""

    def _build(self, counts=(50, 120, 300)):
        rng = np.random.default_rng(0)

        def event(n):
            return {
                'log_mass_1': jnp.asarray(rng.uniform(np.log(6), np.log(40), n)),
                'redshift': jnp.asarray(rng.uniform(0.05, 1.2, n)),
                'log_prior': jnp.zeros(n),
            }

        posteriors = [event(n) for n in counts]
        n_inj = 4000
        injections = {
            'log_mass_1': jnp.asarray(rng.uniform(np.log(6), np.log(40), n_inj)),
            'redshift': jnp.asarray(rng.uniform(0.05, 1.2, n_inj)),
            'log_prior': jnp.zeros(n_inj),
            'total_generated': 1e6,
            'analysis_time': 1.0,
        }
        return PixelPopData(
            name='ragged_test',
            posteriors=posteriors,
            injections=injections,
            pixelpop_parameters=['log_mass_1', 'redshift'],
            other_parameters=[],
            bins=10,
            minima={'log_mass_1': float(np.log(3)), 'redshift': 0.0},
            maxima={'log_mass_1': float(np.log(100)), 'redshift': 2.0},
        )

    def test_ragged_construction(self):
        counts = (50, 120, 300)
        pp = self._build(counts)

        self.assertEqual(pp.Nobs, len(counts))
        self.assertEqual(len(pp.event_bins), len(counts))

        for ii, n in enumerate(counts):
            event = pp.posteriors[ii]
            self.assertIn('ln_dVTc', event)
            self.assertEqual(event['ln_dVTc'].shape, (n,))
            self.assertEqual(event['log_prior'].shape, (n,))
            # one bin-index array per pixelpop dimension, each with this event's count
            self.assertEqual(pp.event_bins[ii][0].shape, (n,))
            self.assertEqual(len(pp.event_bins[ii]), pp.dimension)
