import unittest
import jax.numpy as jnp
import pixelpop
from pixelpop.utils.data import (
    convert_m1q_to_lm1m2,
    convert_m1_to_lm1,
    convert_m1m2_to_lm1lm2,
    clean_par,
    check_bins,
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
