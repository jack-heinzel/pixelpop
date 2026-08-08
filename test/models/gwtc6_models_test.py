"""
Numerical tests for the GWTC-6 parametric models.

Every model in ``gwtc6_default`` / ``gwtc6_fms_default`` is checked against the
reference implementation it is meant to reproduce -- ``gwtc6_population_models``
where one exists, otherwise ``gwpopulation`` -- plus normalization, support and
differentiability checks that the references cannot provide.

``gwtc6_population_models`` is optional; the tests that need it skip if it is
absent. The suite runs in whatever precision is configured, which is float32
unless something has called ``gwpopulation.set_backend``, so tolerances are set
for float32. :func:`test_joint_float32_matches_float64` guards the precision
claim itself.

Machinery around the default sets (hyperparameter ordering, priors, merge) is
tested separately in ``gwtc_defaults_test.py``.
"""
import json
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pixelpop.models import (
    BrokenPowerlawPlusTwoPeaks_PrimaryMass,
    GWTC_DEFAULTS,
    PowerlawRedshiftPsi,
    SmoothedPowerlaw_MassRatio,
    TripleBrokenPowerLaw,
    TripleBrokenPowerlawPlusTwoPeaks_PrimaryMass,
    gwtc6_default,
    gwtc6_fms_default,
    tilt_model,
    two_gaussian_spin,
    two_gaussian_spin_fms,
)

# Both sides run in the same precision, so these are float32 round-off budgets,
# not statements about the models' accuracy. The mass models additionally
# normalize on a different grid from the reference, hence the looser bound.
RTOL = 1e-5
MASS_RTOL = 2e-4
ATOL = 1e-6

# One physically sensible point in the GWTC-6 full-spectrum hyperparameter space,
# keyed by the names in gwtc6_fms_default.hyperparameters.
HYPERS = dict(
    alpha_1=3.5, alpha_2=1.5, alpha_3=4.0, mlow_1=1.2, break_mass_1=5.0,
    break_mass_2=40.0, delta_m_1=1.0, lam_fractions=(0.94, 0.03, 0.03),
    mpp_1=10.0, sigpp_1=2.0, mpp_2=33.0, sigpp_2=4.0, mmax=300.0,
    gaussian_mass_maximum=350.0, break_mass=35.0,
    beta=1.1, mlow_2=1.2, delta_m_2=1.2,
    mu_1_chi=0.1, mu_2_chi=0.4, sigma_1_chi=0.1, sigma_2_chi=0.2,
    lamb_chi_1=0.5, lamb_chi_2=0.5, amax=1.0,
    mu_spin=0.5, sigma_spin=1.0, xi_spin=0.5,
    lamb=2.7, max_z=1.9,
)


@pytest.fixture(scope='module')
def g6():
    return pytest.importorskip('gwtc6_population_models')


@pytest.fixture(scope='module')
def dataset():
    """A rectangular block spanning the full mass spectrum, NSs included."""
    rng = np.random.default_rng(0)
    n = 4000
    m1 = rng.uniform(1.05, 250., n)
    q = rng.uniform(0.02, 1., n)
    return {
        'mass_1': jnp.asarray(m1),
        'log_mass_1': jnp.asarray(np.log(m1)),
        'mass_ratio': jnp.asarray(q),
        'mass_2': jnp.asarray(m1 * q),
        'a_1': jnp.asarray(rng.uniform(1e-4, 1 - 1e-4, n)),
        'a_2': jnp.asarray(rng.uniform(1e-4, 1 - 1e-4, n)),
        'cos_tilt_1': jnp.asarray(rng.uniform(-0.999, 0.999, n)),
        'cos_tilt_2': jnp.asarray(rng.uniform(-0.999, 0.999, n)),
        'redshift': jnp.asarray(rng.uniform(1e-3, 1.85, n)),
    }


def assert_matches(log_prob, reference, rtol=RTOL, atol=ATOL):
    """Compare a pixelpop log density against a reference linear density,
    on the reference's support."""
    log_prob, reference = np.asarray(log_prob), np.asarray(reference)
    ok = np.isfinite(log_prob) & np.isfinite(reference) & (reference > 1e-10)
    assert ok.sum() > 0.2 * reference.size, "reference has almost no support here"
    np.testing.assert_allclose(np.exp(log_prob[ok]), reference[ok],
                               rtol=rtol, atol=atol)


def integrate(log_prob, x):
    return np.trapezoid(np.exp(np.clip(np.asarray(log_prob), -700, None)), x)


# ---------------------------------------------------------------------------
# Primary mass
# ---------------------------------------------------------------------------

BBH_MASS_CASES = [
    dict(alpha_1=1.1, alpha_2=3.0, mmin=4.9, break_mass=35., delta_m_1=3.,
         lam_fractions=(0.5, 0.4, 0.1), mpp_1=10., sigpp_1=1.5, mpp_2=35., sigpp_2=5.),
    dict(alpha_1=3.0, alpha_2=5.5, mmin=3.5, break_mass=25., delta_m_1=1.,
         lam_fractions=(0.9, 0.05, 0.05), mpp_1=8., sigpp_1=1., mpp_2=45., sigpp_2=8.),
]

FMS_MASS_CASES = [
    dict(alpha_1=3.5, alpha_2=1.5, alpha_3=4.0, mmin=1.2, break_mass_1=5.,
         break_mass_2=40., delta_m_1=1.0, lam_fractions=(0.94, 0.03, 0.03),
         mpp_1=10., sigpp_1=2., mpp_2=33., sigpp_2=4.),
    dict(alpha_1=1.0, alpha_2=2.5, alpha_3=6.0, mmin=1.0, break_mass_1=3.5,
         break_mass_2=25., delta_m_1=0.5, lam_fractions=(0.6, 0.2, 0.2),
         mpp_1=8., sigpp_1=1.5, mpp_2=35., sigpp_2=5.),
    dict(alpha_1=-2.0, alpha_2=4.0, alpha_3=2.0, mmin=2.0, break_mass_1=12.,
         break_mass_2=45., delta_m_1=4.0, lam_fractions=(0.8, 0.15, 0.05),
         mpp_1=15., sigpp_1=3., mpp_2=50., sigpp_2=8.),
]


@pytest.mark.parametrize("case", BBH_MASS_CASES)
def test_bbh_primary_mass_matches_gwtc6(g6, case):
    """The GWTC-6 BBH configuration of BrokenPowerlawPlusTwoPeaks_PrimaryMass --
    note gaussian_mass_maximum=350, not pixelpop's default of 100."""
    from gwtc6_population_models.mass import (
        TwoPeakBrokenPowerLawSmoothedMassDistribution,
    )

    model = TwoPeakBrokenPowerLawSmoothedMassDistribution()
    assert model.kwargs['gaussian_mass_maximum'] == 350
    lam_0, lam_1, _ = case['lam_fractions']
    m = np.linspace(case['mmin'] + 0.15, 295., 4000)

    reference = model.p_m1(
        {'mass_1': jnp.asarray(m)}, alpha_1=case['alpha_1'], alpha_2=case['alpha_2'],
        mmin=case['mmin'], mmax=300., break_mass=case['break_mass'],
        delta_m=case['delta_m_1'], lam_0=lam_0, lam_1=lam_1, mpp_1=case['mpp_1'],
        sigpp_1=case['sigpp_1'], mpp_2=case['mpp_2'], sigpp_2=case['sigpp_2'],
        **model.kwargs,
    )
    mine = BrokenPowerlawPlusTwoPeaks_PrimaryMass(
        {'mass_1': jnp.asarray(m)}, mmax=300., gaussian_mass_maximum=350., **case
    )
    assert_matches(mine, reference, rtol=MASS_RTOL)


@pytest.mark.parametrize("case", FMS_MASS_CASES)
def test_fms_primary_mass_matches_gwtc6(g6, case):
    from gwtc6_population_models.mass import (
        TwoPeakThreeBrokenPowerLawSmoothedMassDistribution,
    )

    model = TwoPeakThreeBrokenPowerLawSmoothedMassDistribution()
    lam_0, lam_1, _ = case['lam_fractions']
    m = np.linspace(1.05, 295., 6000)

    reference = model.p_m1(
        {'mass_1': jnp.asarray(m)}, alpha_1=case['alpha_1'], alpha_2=case['alpha_2'],
        alpha_3=case['alpha_3'], mmin=case['mmin'], mmax=300.,
        delta_m=case['delta_m_1'], break_mass_1=case['break_mass_1'],
        break_mass_2=case['break_mass_2'], lam_0=lam_0, lam_1=lam_1,
        mpp_1=case['mpp_1'], sigpp_1=case['sigpp_1'], mpp_2=case['mpp_2'],
        sigpp_2=case['sigpp_2'], **model.kwargs,
    )
    mine = TripleBrokenPowerlawPlusTwoPeaks_PrimaryMass({'mass_1': jnp.asarray(m)}, **case)
    assert_matches(mine, reference, rtol=MASS_RTOL)


@pytest.mark.parametrize("case", BBH_MASS_CASES + FMS_MASS_CASES)
def test_primary_mass_is_normalized(case):
    """Both primary mass models integrate to 1 over the mass range they normalize on."""
    fms = 'alpha_3' in case
    model = (TripleBrokenPowerlawPlusTwoPeaks_PrimaryMass if fms
             else BrokenPowerlawPlusTwoPeaks_PrimaryMass)
    kwargs = dict(gaussian_mass_maximum=350.) if not fms else {}
    m = np.linspace(1.0 if fms else 3.0, 300., 30000)
    logp = model({'mass_1': jnp.asarray(m)}, mmax=300., **kwargs, **case)
    assert integrate(logp, m) == pytest.approx(1.0, abs=5e-3)


@pytest.mark.parametrize("case", FMS_MASS_CASES)
def test_primary_mass_log_jacobian(case):
    """log_mass_1 parameterization differs from mass_1 by exactly log(m1)."""
    m = np.linspace(1.05, 250., 3000)
    linear = np.asarray(TripleBrokenPowerlawPlusTwoPeaks_PrimaryMass(
        {'mass_1': jnp.asarray(m)}, **case))
    log = np.asarray(TripleBrokenPowerlawPlusTwoPeaks_PrimaryMass(
        {'log_mass_1': jnp.asarray(np.log(m))}, **case))
    ok = np.isfinite(linear) & (linear > -50)
    np.testing.assert_allclose(log[ok] - linear[ok], np.log(m)[ok], rtol=1e-4, atol=1e-4)


TRIPLE_PL_CASES = [
    # alpha_1, alpha_2, alpha_3, mmin, mmax, break_1, break_2
    # Avoid alpha_i == -1: gwtc6 negates it and gwpopulation's powerlaw carries an
    # `alpha != 1` guard on the numpy backend, so the *reference* refuses to evaluate
    # there. pixelpop handles it (see "powerlaw_close_to_1" in gwpop_models_test.py);
    # there is simply nothing to compare against.
    (1.1, 3.0, 5.0, 1.0, 300., 5., 35.),
    (-1.5, 2.0, 4.0, 2.0, 300., 10., 40.),
    (3.5, 0.5, 6.0, 1.0, 250., 4., 20.),
]


@pytest.mark.parametrize("case", TRIPLE_PL_CASES)
def test_triple_broken_powerlaw_matches_gwtc6(g6, case):
    from gwtc6_population_models.mass import triple_power_law_primary_mass

    a1, a2, a3, mmin, mmax, b1, b2 = case
    m = np.linspace(1.2, 290., 4000)
    reference = triple_power_law_primary_mass(m, a1, a2, a3, mmin, mmax, b1, b2)
    # pixelpop takes slopes, gwtc6 takes the negated indices
    mine = TripleBrokenPowerLaw(jnp.asarray(m), -a1, -a2, -a3, mmin, mmax, b1, b2)
    assert_matches(mine, reference)


@pytest.mark.parametrize("case", TRIPLE_PL_CASES)
def test_triple_broken_powerlaw_is_continuous(case):
    """The continuity corrections must leave no jump at either break."""
    a1, a2, a3, mmin, mmax, b1, b2 = case
    eps = 1e-5
    for brk in (b1, b2):
        x = jnp.asarray([brk * (1 - eps), brk * (1 + eps)])
        lo, hi = np.asarray(TripleBrokenPowerLaw(x, -a1, -a2, -a3, mmin, mmax, b1, b2))
        assert np.exp(hi) == pytest.approx(np.exp(lo), rel=1e-3)


# ---------------------------------------------------------------------------
# Mass ratio
# ---------------------------------------------------------------------------

MASS_RATIO_CASES = [
    # beta, mmin, delta_m, m1
    (1.1, 5.0, 3.0, 30.),      # BBH
    (2.0, 4.0, 5.0, 80.),
    (1.1, 5.0, 3.0, 10.),
    (1.1, 1.2, 1.0, 150.),     # above the old model's fiducial m1 grid
    (1.1, 1.2, 0.5, 2.5),      # NSBH -- PowerlawPlusPeak_MassRatio is off by 0.57 here
    (1.1, 1.0, 0.2, 1.35),     # BNS  -- ...and by 3.5 here
]


def _reference_p_q(grid, m1v, q, beta, mmin, delta_m):
    """gwpopulation's BaseSmoothedMassDistribution.p_q, evaluated statelessly.

    The stock p_q caches an interpolant built with ``to_numpy(nodes)`` closed over
    the data masses, which neither survives a jit trace nor serves two data shapes.
    All it does is map the per-m1s normalizations onto the data, which ``jnp.interp``
    does statelessly.
    """
    from gwpopulation.utils import powerlaw as gwpop_powerlaw, trapezoid

    m1 = np.full_like(q, m1v)
    prob = gwpop_powerlaw(jnp.asarray(q), beta, 1, mmin / m1)
    prob = prob * grid.smoothing(m1 * q, mmin=mmin, mmax=m1, delta_m=delta_m)
    on_grid = gwpop_powerlaw(grid.qs_grid, beta, 1, mmin / grid.m1s_grid)
    on_grid = on_grid * grid.smoothing(grid.m1s_grid * grid.qs_grid, mmin=mmin,
                                       mmax=grid.m1s_grid, delta_m=delta_m)
    norms = jnp.nan_to_num(trapezoid(on_grid, grid.qs, axis=0))
    return np.asarray(
        prob / jnp.clip(jnp.interp(jnp.log(m1), jnp.log(grid.m1s), norms), 1e-30)
    )


@pytest.mark.parametrize("case", MASS_RATIO_CASES)
def test_mass_ratio_matches_gwtc6(g6, case):
    from gwtc6_population_models.mass import (
        TwoPeakThreeBrokenPowerLawSmoothedMassDistribution,
    )

    beta, mmin, delta_m, m1v = case
    q = np.linspace(0.005, 1.0, 4000)
    reference = _reference_p_q(
        TwoPeakThreeBrokenPowerLawSmoothedMassDistribution(), m1v, q, beta, mmin, delta_m
    )
    data = {'log_mass_1': jnp.asarray(np.full_like(q, np.log(m1v))),
            'mass_ratio': jnp.asarray(q)}
    assert_matches(SmoothedPowerlaw_MassRatio(data, beta, mmin, delta_m),
                   reference, rtol=5e-3)


@pytest.mark.parametrize("case", MASS_RATIO_CASES)
def test_mass_ratio_is_normalized(case):
    """The reason this model exists: it integrates to 1 in q at every m1, including
    below m1 = 2 where PowerlawPlusPeak_MassRatio's fiducial grid does not reach."""
    beta, mmin, delta_m, m1v = case
    q = np.linspace(1e-4, 1.0, 20000)
    data = {'log_mass_1': jnp.asarray(np.full_like(q, np.log(m1v))),
            'mass_ratio': jnp.asarray(q)}
    integral = integrate(SmoothedPowerlaw_MassRatio(data, beta, mmin, delta_m), q)
    assert integral == pytest.approx(1.0, abs=5e-3)


def test_mass_ratio_vanishes_below_mmin():
    """m1 < mmin, and q below mmin/m1, both give vanishing probability rather than a
    spurious finite value from dividing two sentinels.

    The threshold is -100, not the -INF sentinel: m_smoother clips
    (m - mmin)/delta_m to EDGE_FRACTION rather than branching, so below mmin it
    floors at about -1/EDGE_FRACTION = -1000 and keeps falling from there. That
    underflows to exactly zero probability, and stays far enough below
    MASS_RATIO_LOG_NORM_FLOOR that subtracting the floored norm cannot lift it back
    into a spurious spike.
    """
    q = jnp.asarray([0.1, 0.5, 0.9])
    below = {'log_mass_1': jnp.log(jnp.full(3, 2.0)), 'mass_ratio': q}
    out = np.asarray(SmoothedPowerlaw_MassRatio(below, 1.1, 5., 1.))
    assert np.all(out < -100.) and not np.any(np.isnan(out))
    assert np.all(np.exp(np.clip(out, -700, None)) == 0.)

    # m1 = 20, mmin = 5 -> q < 0.25 is out of support, q = 0.5, 0.9 are in
    above = {'log_mass_1': jnp.log(jnp.full(3, 20.0)), 'mass_ratio': q}
    out = np.asarray(SmoothedPowerlaw_MassRatio(above, 1.1, 5., 1.))
    assert out[0] < -100.
    assert np.all(out[1:] > -10.) and np.all(np.isfinite(out[1:]))


# ---------------------------------------------------------------------------
# Component spins
# ---------------------------------------------------------------------------

SPIN_CASES = [
    dict(mu_1_chi=.1, mu_2_chi=.4, sigma_1_chi=.1, sigma_2_chi=.2,
         lamb_chi_1=.5, lamb_chi_2=.5),
    dict(mu_1_chi=.02, mu_2_chi=.8, sigma_1_chi=.05, sigma_2_chi=.3,
         lamb_chi_1=.9, lamb_chi_2=.1),
]


@pytest.mark.parametrize("case", SPIN_CASES)
def test_two_gaussian_spin_matches_gwtc6(g6, dataset, case):
    from gwtc6_population_models.spin import spin_magnitude_two_gaussians_BBH

    assert_matches(two_gaussian_spin(dataset, **case),
                   spin_magnitude_two_gaussians_BBH(dataset, amax=1., **case))


@pytest.mark.parametrize("case", SPIN_CASES)
def test_two_gaussian_spin_fms_matches_gwtc6(g6, dataset, case):
    from gwtc6_population_models.spin import spin_magnitude_two_gaussians_CBC

    reference = np.asarray(spin_magnitude_two_gaussians_CBC(dataset, amax=1., **case))
    mine = np.asarray(two_gaussian_spin_fms(dataset, **case))

    assert (reference <= 0).sum() > 0, "expected the NS cap to zero some samples"
    assert_matches(mine, reference)
    # the support must agree exactly, not just the values inside it
    np.testing.assert_array_equal(reference <= 0, mine < -1e9)


def test_ns_cap_is_applied_per_component():
    """A neutron-star component may not spin above NS_amax, whatever its partner's
    mass is."""
    data = {
        'a_1': jnp.asarray([0.6, 0.6, 0.2, 0.2]),
        'a_2': jnp.asarray([0.2, 0.6, 0.6, 0.2]),
        'mass_1': jnp.asarray([1.4, 1.4, 30., 1.4]),   # primary is an NS except [2]
        'mass_2': jnp.asarray([1.3, 30., 1.3, 1.3]),
    }
    data['mass_ratio'] = data['mass_2'] / data['mass_1']
    out = np.asarray(two_gaussian_spin_fms(
        data, mu_1_chi=.1, mu_2_chi=.4, sigma_1_chi=.1, sigma_2_chi=.2,
        lamb_chi_1=.5, lamb_chi_2=.5))

    # rows 0-2 each put a spin of 0.6 on a sub-2.5 Msun component -> excluded
    assert np.all(out[:3] < -1e9)
    assert np.isfinite(out[3]) and out[3] > -1e9


def test_two_gaussian_spin_without_cap_ignores_mass(dataset):
    """The BBH model must not depend on the masses at all."""
    case = SPIN_CASES[0]
    spins_only = {k: dataset[k] for k in ('a_1', 'a_2')}
    np.testing.assert_array_equal(
        np.asarray(two_gaussian_spin(dataset, **case)),
        np.asarray(two_gaussian_spin(spins_only, **case)),
    )


@pytest.mark.parametrize("case", SPIN_CASES)
def test_two_gaussian_spin_marginal_is_normalized(case):
    """The single-component branch (used when saving popsummary marginals) is a
    density in a over [0, amax]."""
    a = np.linspace(1e-6, 1 - 1e-6, 40000)
    logp = two_gaussian_spin({'a': jnp.asarray(a)}, **case)
    assert integrate(logp, a) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("case", SPIN_CASES)
def test_two_gaussian_spin_is_separable(case):
    """The joint is p(a_1) p(a_2), so log p(a_1, a_2) - log p(a_1, a_2') must not
    depend on a_1. Independent mixing fractions make this the property to check
    rather than symmetry under swapping a_1 and a_2."""
    a = np.linspace(0.01, 0.99, 500)
    both = np.asarray(two_gaussian_spin(
        {'a_1': jnp.asarray(a), 'a_2': jnp.full(a.shape, 0.3)}, **case))
    other = np.asarray(two_gaussian_spin(
        {'a_1': jnp.asarray(a), 'a_2': jnp.full(a.shape, 0.7)}, **case))
    difference = both - other
    np.testing.assert_allclose(difference, difference[0], rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Tilt and redshift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mu,sig,zeta", [(0.5, 1.0, 0.5), (1.0, 0.6, 0.8),
                                         (-0.3, 2.0, 0.2)])
def test_tilt_matches_gwpopulation(dataset, mu, sig, zeta):
    """GWTC-6 uses the free-mean joint tilt model, i.e. pixelpop's tilt_model, not
    the per-component tilt_iid that gwtc4_default uses."""
    from gwpopulation.models.spin import iid_spin_orientation_gaussian_isotropic

    assert gwtc6_default.models['t'] is tilt_model
    reference = iid_spin_orientation_gaussian_isotropic(
        dataset, xi_spin=zeta, sigma_spin=sig, mu_spin=mu)
    assert_matches(tilt_model(dataset, mu, sig, zeta), reference)


@pytest.mark.parametrize("lamb", [-2.0, 0.0, 2.7, 6.0])
def test_redshift_psi_matches_gwpopulation(dataset, lamb):
    """pixelpop takes the (1+z)^lamb factor only; the comoving-volume element is
    folded into the prior by PixelPopData.preprocess_cosmology."""
    from gwpopulation.models.redshift import PowerLawRedshift

    reference = PowerLawRedshift(z_max=1.9).psi_of_z(dataset['redshift'], lamb=lamb)
    assert_matches(PowerlawRedshiftPsi(dataset, lamb), reference)


def test_redshift_psi_zero_above_max_z():
    z = jnp.asarray([0.5, 1.8, 1.95, 3.0])
    out = np.asarray(PowerlawRedshiftPsi({'redshift': z}, 2.7, max_z=1.9))
    assert np.all(out[:2] > -10.) and np.all(out[2:] < -1e9)


# ---------------------------------------------------------------------------
# The suites as a whole
# ---------------------------------------------------------------------------

# The population parameters that form the joint density; windows and the effective
# spin models are alternatives to these rather than additional factors.
JOINT_PARAMETERS = ['log_mass_1', 'mass_ratio', 'a', 't', 'redshift']


def joint_log_density(defaults, data, hypers=HYPERS):
    """Sum the suite's models, calling each exactly the way probabilistic.py does."""
    total = jnp.zeros_like(data['mass_ratio'])
    for parameter in JOINT_PARAMETERS:
        model = defaults.models[parameter]
        total = total + model(
            data, *[hypers[h] for h in defaults.hyperparameters[parameter]]
        )
    return total


def test_joint_matches_gwtc6(g6, dataset):
    """The whole gwtc6_fms suite against the gwtc6 stack, driven through the real
    hyperparameter lists. This is what catches a mis-ordered hyperparameter list,
    which the per-model tests cannot: they pass arguments by keyword."""
    from gwtc6_population_models.mass import (
        TwoPeakThreeBrokenPowerLawSmoothedMassDistribution,
    )
    from gwtc6_population_models.spin import spin_magnitude_two_gaussians_CBC
    from gwpopulation.models.redshift import PowerLawRedshift
    from gwpopulation.models.spin import iid_spin_orientation_gaussian_isotropic

    h = HYPERS
    grid = TwoPeakThreeBrokenPowerLawSmoothedMassDistribution()
    m1 = np.asarray(dataset['mass_1'])

    p_m1 = grid.p_m1(
        {'mass_1': dataset['mass_1']}, alpha_1=h['alpha_1'], alpha_2=h['alpha_2'],
        alpha_3=h['alpha_3'], mmin=h['mlow_1'], mmax=h['mmax'], delta_m=h['delta_m_1'],
        break_mass_1=h['break_mass_1'], break_mass_2=h['break_mass_2'],
        lam_0=h['lam_fractions'][0], lam_1=h['lam_fractions'][1], mpp_1=h['mpp_1'],
        sigpp_1=h['sigpp_1'], mpp_2=h['mpp_2'], sigpp_2=h['sigpp_2'], **grid.kwargs,
    )
    # pixelpop works in log_mass_1, so the reference needs the |J| = m1 jacobian
    reference = (
        np.asarray(p_m1) * m1
        * np.asarray(spin_magnitude_two_gaussians_CBC(
            dataset, amax=h['amax'], mu_1_chi=h['mu_1_chi'], mu_2_chi=h['mu_2_chi'],
            sigma_1_chi=h['sigma_1_chi'], sigma_2_chi=h['sigma_2_chi'],
            lamb_chi_1=h['lamb_chi_1'], lamb_chi_2=h['lamb_chi_2']))
        * np.asarray(iid_spin_orientation_gaussian_isotropic(
            dataset, xi_spin=h['xi_spin'], sigma_spin=h['sigma_spin'],
            mu_spin=h['mu_spin']))
        * np.asarray(PowerLawRedshift(z_max=h['max_z']).psi_of_z(
            dataset['redshift'], lamb=h['lamb']))
    )
    q = np.asarray(dataset['mass_ratio'])
    p_q = np.array([
        _reference_p_q(grid, m1v, np.array([qv]), h['beta'], h['mlow_2'],
                       h['delta_m_2'])[0]
        for m1v, qv in zip(m1[:300], q[:300])
    ])
    reference = reference[:300] * p_q

    mine = np.asarray(joint_log_density(gwtc6_fms_default, dataset))[:300]
    assert_matches(mine, reference, rtol=5e-3)


@pytest.mark.parametrize("catalog", ['gwtc6', 'gwtc6_fms'])
def test_suite_is_jittable_and_differentiable(catalog, dataset):
    """Every model in the suite must jit and grad cleanly w.r.t. every scalar
    hyperparameter -- NUTS needs the gradient, not just the value."""
    defaults = GWTC_DEFAULTS[catalog]
    names = [h for p in JOINT_PARAMETERS for h in defaults.hyperparameters[p]]
    names = [n for n in dict.fromkeys(names) if np.ndim(HYPERS[n]) == 0]

    def loss(*values):
        hypers = dict(HYPERS, **dict(zip(names, values)))
        out = joint_log_density(defaults, dataset, hypers)
        return jnp.sum(jnp.where(jnp.isfinite(out), out, 0.0))

    args = [jnp.asarray(HYPERS[n], dtype=float) for n in names]
    grads = jax.jit(jax.grad(loss, argnums=tuple(range(len(names)))))(*args)
    for name, g in zip(names, grads):
        g = np.asarray(g)
        assert not np.any(np.isnan(g)), f"{catalog}: NaN gradient w.r.t. {name}"
        assert not np.any(np.isinf(g)), f"{catalog}: inf gradient w.r.t. {name}"


def _precision_data(n=1500):
    """Plain-numpy version of the `dataset` fixture, so the float64 subprocess can be
    handed exactly the same numbers through a .npz instead of re-deriving them."""
    rng = np.random.default_rng(0)
    m1 = rng.uniform(1.05, 250., n)
    q = rng.uniform(0.02, 1., n)
    return {
        'mass_1': m1, 'log_mass_1': np.log(m1), 'mass_ratio': q, 'mass_2': m1 * q,
        'a_1': rng.uniform(1e-4, 1 - 1e-4, n), 'a_2': rng.uniform(1e-4, 1 - 1e-4, n),
        'cos_tilt_1': rng.uniform(-0.999, 0.999, n),
        'cos_tilt_2': rng.uniform(-0.999, 0.999, n),
        'redshift': rng.uniform(1e-3, 1.85, n),
    }


def test_joint_float32_matches_float64(tmp_path):
    """The premise of the log-space rewrite: float32 is enough.

    jax_enable_x64 is a process-global flag that cannot be toggled reliably once
    anything has been traced, so the float64 reference is computed in a subprocess.
    """
    script = textwrap.dedent(f"""
        import json, os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import jax
        jax.config.update('jax_enable_x64', True)
        import numpy as np, jax.numpy as jnp
        from pixelpop.models import gwtc6_fms_default as D

        hypers = json.loads({json.dumps(HYPERS)!r})
        parameters = json.loads({json.dumps(JOINT_PARAMETERS)!r})
        data = {{k: jnp.asarray(v) for k, v in
                 np.load({str(tmp_path / 'data.npz')!r}).items()}}
        assert jnp.zeros(1).dtype == jnp.float64

        total = jnp.zeros_like(data['mass_ratio'])
        for p in parameters:
            total = total + D.models[p](
                data, *[hypers[h] for h in D.hyperparameters[p]])
        np.save({str(tmp_path / 'f64.npy')!r}, np.asarray(total, dtype=np.float64))
    """)
    np.savez(tmp_path / 'data.npz',
             **{k: np.asarray(v, dtype=np.float64) for k, v in _precision_data().items()})
    # The parent already has XLA's thread pools up, so cap the child's hard --
    # otherwise it trips the per-user thread limit ("pthread_create() failed").
    # JAX_PLATFORMS=cpu also skips the CUDA plugin probe, which is pointless here.
    env = dict(os.environ,
               JAX_PLATFORMS='cpu',
               XLA_FLAGS=('--xla_force_host_platform_device_count=1 '
                          '--xla_cpu_multi_thread_eigen=false'),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NPROC='1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    result = subprocess.run([sys.executable, '-c', script],
                            capture_output=True, text=True, env=env)

    if result.returncode != 0:
        # A thread-starved box (low `ulimit -u`, busy machine) cannot start a second
        # XLA process at all. That is an environment limit, not a defect -- skip
        # rather than report a failure the code cannot fix.
        starved = ('pthread_create' in result.stderr
                   or 'Resource temporarily unavailable' in result.stderr)
        if starved:
            pytest.skip("cannot spawn the float64 subprocess: the machine is out of "
                        "threads (check `ulimit -u`). Run this test on its own.")
        pytest.fail(f"float64 subprocess failed:\n{result.stderr[-3000:]}")

    reference = np.load(tmp_path / 'f64.npy')

    data = {k: jnp.asarray(v) for k, v in _precision_data().items()}
    mine = np.asarray(joint_log_density(gwtc6_fms_default, data), dtype=np.float64)

    # in-support samples only: out of support both sides sit on the -INF sentinel,
    # where float32 spacing is ~1024 and a comparison is meaningless
    ok = (reference > -50) & (mine > -50)
    assert ok.sum() > 0.3 * reference.size
    np.testing.assert_allclose(mine[ok], reference[ok], rtol=0, atol=5e-3)
