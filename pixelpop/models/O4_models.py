"""
O4-era (GWTC-4 onwards) population models.

These are the named population models introduced for the O4 catalogs, kept apart
from the O3/GWTC-3 models and the generic primitives so that the catalog default
sets in :mod:`~pixelpop.models.gwtc_defaults` have an obvious home. Everything
here is re-exported by :mod:`~pixelpop.models.gwpop_models`, so existing imports
are unaffected.

Contents
--------
Primary mass
    :func:`BrokenPowerlawPlusTwoPeaks_PrimaryMass` (GWTC-4 default),
    :func:`TripleBrokenPowerlawPlusTwoPeaks_PrimaryMass` (GWTC-6 full mass
    spectrum), :func:`WrongOrderSmoothed_BrokenPowerlawPlusTwoPeaks_PrimaryMass`.
Spin magnitude
    :func:`iid_normal_spin`, :func:`iid_normal_spin_fms`,
    :func:`two_gaussian_spin`, :func:`two_gaussian_spin_fms` (GWTC-6).
Spin tilt
    :func:`tilt_iid`.

All densities are returned in log space.
"""
import jax.numpy as jnp
import jax.scipy.special as scs

from .base_models import (
    INF,
    BrokenPowerLaw,
    TripleBrokenPowerLaw,
    m_smoother,
    trunc_gaussian,
)

# Normalization grid for the full-mass-spectrum primary mass model. Log spaced,
# unlike the BBH models' linear grid: the full spectrum has to resolve structure
# at ~1.4 Msun (the NS range) and at ~35 Msun with the same grid, and a linear
# 2000-point grid over [1, 300] has 0.15 Msun steps -- roughly 10% resolution at
# the NS peak. Log spacing gives ~0.004 Msun there for the same cost.
FMS_GRID_MINIMUM = 1.0
FMS_GRID_MAXIMUM = 300.0
FMS_GRID_POINTS = 2000

# The neutron-star spin cap applied by the GWTC-6 full-spectrum spin model.
NS_SPIN_MAXIMUM = 0.4
NS_MASS_MAXIMUM = 2.5

# Normalization grid for SmoothedPowerlaw_MassRatio. The m1 axis is log spaced so the
# norm can be interpolated down into the neutron-star range; the q axis is rescaled to
# each m1's own support [mmin/m1, 1], since a grid with a fixed q floor either starves
# the small-m1 end (where the support is a sliver near q = 1) or the turn-on at large
# m1. 1000 q points with the trapezoid rule holds the normalization to <1e-3.
MASS_RATIO_Q_POINTS = 1000
MASS_RATIO_M1_MINIMUM = 1.0
MASS_RATIO_M1_MAXIMUM = 300.0
MASS_RATIO_M1_POINTS = 500
# Grid points below mmin have no valid q at all, so their norm is the -INF sentinel.
# Floor it well above -INF so it cannot dominate the subtraction, or leak a 1e10 into
# the interpolation for an m1 just above mmin.
MASS_RATIO_LOG_NORM_FLOOR = -100.


def _primary_mass(data):
    """Return ``(m1, log_jacobian)``, where the jacobian is nonzero iff the data
    is parameterized in log mass."""
    if isinstance(data, dict):
        if 'log_mass_1' in data:
            return jnp.exp(data['log_mass_1']), data['log_mass_1']
        return data['mass_1'], 0.0
    return data, 0.0


def _component_masses(data):
    """Return ``(m1, m2)`` from whichever mass parameterization ``data`` carries."""
    m1, _ = _primary_mass(data)
    if 'log_mass_2' in data:
        # The lm1lm2 parameterization drops the linear 'mass_2' key.
        return m1, jnp.exp(data['log_mass_2'])
    if 'mass_2' in data:
        return m1, data['mass_2']
    if 'mass_ratio' in data:
        return m1, m1 * data['mass_ratio']
    raise KeyError(
        "Expected one of 'log_mass_2', 'mass_2' or 'mass_ratio' in data to "
        "determine the secondary mass."
    )


def BrokenPowerlawPlusTwoPeaks_PrimaryMass(
    data, alpha_1, alpha_2, mmin, break_mass, delta_m_1,
    lam_fractions, mpp_1, sigpp_1, mpp_2, sigpp_2,
    mmax=300., gaussian_mass_maximum=100.):
    """
    Primary mass distribution: broken power-law + two Gaussian peaks.

    Implements the default GWTC-4.0 primary mass population model:
    a mixture of (1) a smoothed broken power-law, and (2–3) two
    truncated Gaussians representing additional features.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict with key 'mass_1' or 'log_mass_1',
        or a direct array of primary masses.
    alpha_1 : float
        Low-mass slope of the power-law.
    alpha_2 : float
        High-mass slope of the power-law.
    mmin : float
        Minimum primary mass cutoff.
    break_mass : float
        Break mass separating the two slopes.
    delta_m_1 : float
        Smoothing width at the low-mass cutoff.
    lam_fractions : tuple of floats
        Mixture fractions (lam_0, lam_1, lam_2) for
        {power-law, first Gaussian, second Gaussian}.
    mpp_1 : float
        Mean of the first Gaussian peak.
    sigpp_1 : float
        Std. deviation of the first Gaussian peak.
    mpp_2 : float
        Mean of the second Gaussian peak.
    sigpp_2 : float
        Std. deviation of the second Gaussian peak.
    mmax : float, optional
        Maximum primary mass cutoff (default 300).
    gaussian_mass_maximum : float, optional
        Upper truncation for Gaussian peaks (default 100).
        Note that the GWTC-6 configuration of this model uses 350.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the normalized mass distribution.
    """

    isLogMass = True
    if isinstance(data, dict):
        try:
            m1 = jnp.exp(data['log_mass_1'])
        except KeyError:
            isLogMass = False
            m1 = data['mass_1']
    else:
        isLogMass = False
        m1 = data
    lam_0, lam_1, lam_2 = lam_fractions
    break_fraction = (break_mass  - mmin) / (mmax - mmin)

    def _unnorm_bpl2p(m):
        p_pow = BrokenPowerLaw(m, -alpha_1, -alpha_2, mmin, mmax, break_fraction)

        p_norm1 = trunc_gaussian(
            m, mpp_1, sigpp_1, mmin, gaussian_mass_maximum
        )
        p_norm2 = trunc_gaussian(
            m, mpp_2, sigpp_2, mmin, gaussian_mass_maximum
        )
        p = scs.logsumexp(jnp.array([
            jnp.log(lam_0) + p_pow,
            jnp.log(lam_1) + p_norm1,
            jnp.log(lam_2) + p_norm2
            ]), axis=0)

        p += m_smoother(m, mmin, delta_m_1)
        return p

    m1s_test = jnp.linspace(3.0, 300.0, 2000)
    dm1 = m1s_test[1] - m1s_test[0]

    pm1 = _unnorm_bpl2p(m1)
    pm1test = _unnorm_bpl2p(m1s_test)

    pm1 -= scs.logsumexp(pm1test) + jnp.log(dm1) # simple Riemann rule.
    if isLogMass: # include jacobian
        pm1 = pm1 + data['log_mass_1']
    return pm1


def TripleBrokenPowerlawPlusTwoPeaks_PrimaryMass(
    data, alpha_1, alpha_2, alpha_3, mmin, break_mass_1, break_mass_2,
    delta_m_1, lam_fractions, mpp_1, sigpp_1, mpp_2, sigpp_2,
    mmax=300., gaussian_mass_maximum=350.):
    """
    Primary mass distribution: twice-broken power-law + two Gaussian peaks.

    The GWTC-6 full-mass-spectrum primary mass model, i.e. the log-space
    equivalent of ``gwtc6_population_models.mass``'s
    ``TwoPeakThreeBrokenPowerLawSmoothedMassDistribution``. It is
    :func:`BrokenPowerlawPlusTwoPeaks_PrimaryMass` with a third power-law
    segment, which lets a single model span the neutron-star, low-mass-gap and
    black-hole ranges: `alpha_1` governs the NS range below `break_mass_1`,
    `alpha_2` the gap between the breaks, and `alpha_3` the BH range above
    `break_mass_2`.

    Requires ``mmin < break_mass_1 < break_mass_2 < mmax``, and ``mmin`` at or
    above ``FMS_GRID_MINIMUM``.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict with key 'mass_1' or 'log_mass_1',
        or a direct array of primary masses.
    alpha_1 : float
        Power-law slope below the first break.
    alpha_2 : float
        Power-law slope between the two breaks.
    alpha_3 : float
        Power-law slope above the second break.
    mmin : float
        Minimum primary mass cutoff.
    break_mass_1 : float
        Mass at which the first break occurs.
    break_mass_2 : float
        Mass at which the second break occurs.
    delta_m_1 : float
        Smoothing width at the low-mass cutoff.
    lam_fractions : tuple of floats
        Mixture fractions (lam_0, lam_1, lam_2) for
        {power-law, first Gaussian, second Gaussian}.
    mpp_1 : float
        Mean of the first Gaussian peak.
    sigpp_1 : float
        Std. deviation of the first Gaussian peak.
    mpp_2 : float
        Mean of the second Gaussian peak.
    sigpp_2 : float
        Std. deviation of the second Gaussian peak.
    mmax : float, optional
        Maximum primary mass cutoff (default 300).
    gaussian_mass_maximum : float, optional
        Upper truncation for the Gaussian peaks (default 350, matching GWTC-6).

    Returns
    -------
    jnp.ndarray
        Log-probability density of the normalized mass distribution.
    """
    m1, log_jacobian = _primary_mass(data)
    lam_0, lam_1, lam_2 = lam_fractions

    def _unnorm_tbpl2p(m):
        p_pow = TripleBrokenPowerLaw(
            m, -alpha_1, -alpha_2, -alpha_3, mmin, mmax, break_mass_1, break_mass_2
        )
        p_norm1 = trunc_gaussian(m, mpp_1, sigpp_1, mmin, gaussian_mass_maximum)
        p_norm2 = trunc_gaussian(m, mpp_2, sigpp_2, mmin, gaussian_mass_maximum)
        p = scs.logsumexp(jnp.array([
            jnp.log(lam_0) + p_pow,
            jnp.log(lam_1) + p_norm1,
            jnp.log(lam_2) + p_norm2,
            ]), axis=0)

        p += m_smoother(m, mmin, delta_m_1)
        return p

    log_m1s_test = jnp.linspace(
        jnp.log(FMS_GRID_MINIMUM), jnp.log(FMS_GRID_MAXIMUM), FMS_GRID_POINTS
    )
    dlogm1 = log_m1s_test[1] - log_m1s_test[0]
    m1s_test = jnp.exp(log_m1s_test)

    pm1 = _unnorm_tbpl2p(m1)
    # Riemann rule in log mass: int p dm = int p * m dlogm
    pm1test = _unnorm_tbpl2p(m1s_test) + log_m1s_test

    pm1 -= scs.logsumexp(pm1test) + jnp.log(dlogm1)
    return pm1 + log_jacobian  # jacobian is zero unless data is in log mass


def WrongOrderSmoothed_BrokenPowerlawPlusTwoPeaks_PrimaryMass(
    data, alpha_1, alpha_2, mmin, break_mass, delta_m_1,
    lam_fractions, mpp_1, sigpp_1, mpp_2, sigpp_2,
    mmax=300., gaussian_mass_maximum=100.):
    """
    Primary mass distribution: broken power-law + two Gaussian peaks.

    Implements the default GWTC-4.0 primary mass population model:
    a mixture of (1) a smoothed broken power-law, and (2–3) two
    truncated Gaussians representing additional features.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict with key 'mass_1' or 'log_mass_1',
        or a direct array of primary masses.
    alpha_1 : float
        Low-mass slope of the power-law.
    alpha_2 : float
        High-mass slope of the power-law.
    mmin : float
        Minimum primary mass cutoff.
    break_mass : float
        Break mass separating the two slopes.
    delta_m_1 : float
        Smoothing width at the low-mass cutoff.
    lam_fractions : tuple of floats
        Mixture fractions (lam_0, lam_1, lam_2) for
        {power-law, first Gaussian, second Gaussian}.
    mpp_1 : float
        Mean of the first Gaussian peak.
    sigpp_1 : float
        Std. deviation of the first Gaussian peak.
    mpp_2 : float
        Mean of the second Gaussian peak.
    sigpp_2 : float
        Std. deviation of the second Gaussian peak.
    mmax : float, optional
        Maximum primary mass cutoff (default 300).
    gaussian_mass_maximum : float, optional
        Upper truncation for Gaussian peaks (default 100).

    Returns
    -------
    jnp.ndarray
        Log-probability density of the normalized mass distribution.
    """

    isLogMass = True
    if isinstance(data, dict):
        try:
            m1 = jnp.exp(data['log_mass_1'])
        except KeyError:
            isLogMass = False
            m1 = data['mass_1']
    else:
        isLogMass = False
        m1 = data
    lam_0, lam_1, lam_2 = lam_fractions
    break_fraction = (break_mass  - mmin) / (mmax - mmin)
    p_pow = BrokenPowerLaw(m1, -alpha_1, -alpha_2, mmin, mmax, break_fraction)
    p_pow += m_smoother(m1, mmin, delta_m_1)

    p_norm1 = trunc_gaussian(
        m1, mpp_1, sigpp_1, mmin, gaussian_mass_maximum
    )
    p_norm2 = trunc_gaussian(
        m1, mpp_2, sigpp_2, mmin, gaussian_mass_maximum
    )
    pm1 = scs.logsumexp(jnp.array([
        jnp.log(lam_0) + p_pow,
        jnp.log(lam_1) + p_norm1,
        jnp.log(lam_2) + p_norm2
        ]), axis=0)

    # unnormalized, unsmoothed
    m1s_test = jnp.linspace(3.0, 300.0, 2000)
    dm1 = m1s_test[1] - m1s_test[0]
    p_powtest = BrokenPowerLaw(m1s_test, -alpha_1, -alpha_2, mmin, mmax, break_fraction)
    p_powtest += m_smoother(m1s_test, mmin, delta_m_1)

    p_norm1test = trunc_gaussian(
        m1s_test, mpp_1, sigpp_1, mmin, gaussian_mass_maximum
    )
    p_norm2test = trunc_gaussian(
        m1s_test, mpp_2, sigpp_2, mmin, gaussian_mass_maximum
    )
    pm1test = scs.logsumexp(jnp.array([
        jnp.log(lam_0) + p_powtest,
        jnp.log(lam_1) + p_norm1test,
        jnp.log(lam_2) + p_norm2test
        ]), axis=0)
    pm1 -= scs.logsumexp(pm1test) + jnp.log(dm1) # simple Riemann rule.
    if isLogMass: # include jacobian
        pm1 = pm1 + data['log_mass_1']
    return pm1


def SmoothedPowerlaw_MassRatio(data, slope, minimum, delta_m):
    r"""
    Mass-ratio distribution: power law in q with the secondary-mass turn-on.

    .. math::
        p(q | m_1) \propto q^{\beta}\, S(m_1 q \mid m_{\min}, \delta_m)

    normalized over :math:`q \in [0, 1]` separately at each :math:`m_1`. This is the
    GWTC-6 mass-ratio model, i.e. the log-space equivalent of gwpopulation's
    ``BaseSmoothedMassDistribution.p_q``.

    Prefer this over :func:`~pixelpop.models.gwpop_models.PowerlawPlusPeak_MassRatio`,
    which is the same density but normalizes on a BBH-only fiducial grid and
    mis-normalizes by ~3.5x below ``m1 = 2``. That one is kept unchanged because
    the GWTC-3/4/5 default sets use it.

    Parameters
    ----------
    data : dict
        Must contain 'mass_ratio' and either 'mass_1' or 'log_mass_1'.
    slope : float
        Power-law slope on the mass ratio q.
    minimum : float
        Minimum component mass; sets both the turn-on and the q support.
    delta_m : float
        Width of the smoothing region above `minimum`.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the mass-ratio distribution.
    """
    m1, _ = _primary_mass(data)
    q = data['mass_ratio']

    def log_integrand(qq, mm):
        # q^slope on q <= 1, times the turn-on in m2 = m1 q. m_smoother already
        # returns -INF below mmin, so it carries the q >= mmin/m1 edge as well.
        return (jnp.where(qq <= 1., slope * jnp.log(qq), -INF)
                + m_smoother(qq * mm, minimum, delta_m))

    log_m1s = jnp.linspace(jnp.log(MASS_RATIO_M1_MINIMUM),
                           jnp.log(MASS_RATIO_M1_MAXIMUM), MASS_RATIO_M1_POINTS)
    m1s = jnp.exp(log_m1s)

    # q axis rescaled onto each m1's own support so every column resolves the same
    # number of points across [mmin/m1, 1], and mapped quadratically to cluster them
    # at the bottom, where the turn-on lives (it spans only delta_m/m1 in q, which at
    # m1 = 300 is narrower than a uniform step).
    qmin = jnp.clip(minimum / m1s, 0., 1.)
    unit = jnp.linspace(0., 1., MASS_RATIO_Q_POINTS)[:, None]
    qs = qmin[None, :] + (1. - qmin[None, :]) * unit ** 2
    dq = jnp.clip((1. - qmin) / (MASS_RATIO_Q_POINTS - 1), 1e-30)

    # Trapezoid rather than plain Riemann -- the integrand is O(1) at q = 1, so the
    # left-endpoint rule biases the normalization low by ~h/2, about 1% here. The 2*u
    # is d(q)/d(u) for the quadratic map, folded into the quadrature weight so that
    # log(0) never reaches the logsumexp.
    weights = jnp.ones_like(unit).at[0].set(0.5).at[-1].set(0.5) * 2. * unit

    log_norms = scs.logsumexp(
        log_integrand(qs, m1s[None, :]), b=weights, axis=0
    ) + jnp.log(dq)

    log_norms = jnp.clip(log_norms, MASS_RATIO_LOG_NORM_FLOOR)

    # m1 below mmin needs no separate cut: the numerator's m_smoother is ~-1000*delta_m
    # there, which underflows to zero probability on its own. Note that this is a soft
    # floor, not a hard zero -- m_smoother clips m - mmin to `buffer` rather than
    # branching -- so for delta_m << 0.01 a little probability leaks below mmin. That
    # is pixelpop-wide behaviour, and the normalization above integrates only over
    # [mmin/m1, 1], so it deliberately excludes the leak rather than blessing it.
    return log_integrand(q, m1) - jnp.interp(jnp.log(m1), log_m1s, log_norms)


def iid_normal_spin(data, mu, var):
    """
    Truncated normal distribution for spin magnitudes.

    Parameters
    ----------
    data : dict
        Must contain 'a_1' and 'a_2'
    mu : float
        Truncated normal location parameter.
    var : float
        Truncated normal width parameter.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the truncated normal distribution.
    """
    sig = jnp.sqrt(var)
    if 'a' in data and ('a_1' not in data and 'a_2' not in data):
        # just return the marginal, this is used for saving marginals
        return trunc_gaussian(data['a'], mu, sig, 0, 1)

    return trunc_gaussian(data['a_1'], mu, sig, 0, 1) + trunc_gaussian(data['a_2'], mu, sig, 0, 1)

def iid_normal_spin_fms(data, mu, var, NS_amax=NS_SPIN_MAXIMUM, NS_mmax=NS_MASS_MAXIMUM):
    """
    Truncated normal distribution for spin magnitudes. Enforces the truncation to be
    between 0 and 0.4 wherever the mass is less than 2.5 Msun.

    Parameters
    ----------
    data : dict
        Must contain 'a_1' and 'a_2'
    mu : float
        Truncated normal location parameter.
    var : float
        Truncated normal width parameter.
    NS_amax : float
        Maximum spin for neutron stars
    NS_mmax : float
        Maximum mass for neutron stars

    Returns
    -------
    jnp.ndarray
        Log-probability density of the truncated normal distribution.
    """
    sig = jnp.sqrt(var)
    total_prob = jnp.zeros_like(data['a_1'])
    m1, m2 = _component_masses(data)
    regions = {'mass_1': m1, 'mass_2': m2}
    for ii in [1,2]:
        probs = jnp.where(
            regions[f'mass_{ii}'] < NS_mmax,
            trunc_gaussian(data[f'a_{ii}'], mu, sig, 0, NS_amax),
            trunc_gaussian(data[f'a_{ii}'], mu, sig, 0, 1)
            )
        total_prob += probs

    return total_prob


def _two_gaussian_mixture(a, mu_1, sigma_1, mu_2, sigma_2, lamb, amax):
    """One component's spin magnitude density: a two-truncated-Gaussian mixture
    on ``[0, amax]``, mixed with weight ``lamb`` on the first Gaussian."""
    # clip off -inf so logaddexp never sees (-inf, -inf), which would hand a NaN
    # back through the jnp.where in two_gaussian_spin_fms
    p_1 = jnp.clip(trunc_gaussian(a, mu_1, sigma_1, 0., amax), -INF)
    p_2 = jnp.clip(trunc_gaussian(a, mu_2, sigma_2, 0., amax), -INF)
    return jnp.logaddexp(jnp.log(lamb) + p_1, jnp.log1p(-lamb) + p_2)


def two_gaussian_spin(data, mu_1_chi, mu_2_chi, sigma_1_chi, sigma_2_chi,
                      lamb_chi_1, lamb_chi_2, amax=1.):
    """
    Spin magnitude distribution: mixture of two truncated Gaussians.

    The GWTC-6 BBH component-spin magnitude model, i.e. the log-space equivalent
    of ``gwtc6_population_models.spin.spin_magnitude_two_gaussians_BBH``. Both
    components share the same pair of truncated Gaussians on ``[0, amax]`` but
    have independent mixing fractions.

    Parameters
    ----------
    data : dict
        Must contain 'a_1' and 'a_2' (or 'a', for saving marginals).
    mu_1_chi : float
        Mean of the first (low-spin) Gaussian component.
    mu_2_chi : float
        Mean of the second (high-spin) Gaussian component.
    sigma_1_chi : float
        Std. deviation of the first Gaussian component.
    sigma_2_chi : float
        Std. deviation of the second Gaussian component.
    lamb_chi_1 : float
        Weight on the first Gaussian for the primary spin.
    lamb_chi_2 : float
        Weight on the first Gaussian for the secondary spin.
    amax : float, optional
        Maximum spin magnitude (default 1).

    Returns
    -------
    jnp.ndarray
        Log-probability density of the spin magnitude distribution.
    """
    if 'a' in data and ('a_1' not in data and 'a_2' not in data):
        # just return the marginal, this is used for saving marginals. The two
        # components differ only in their mixing fraction; use the primary's.
        return _two_gaussian_mixture(
            data['a'], mu_1_chi, sigma_1_chi, mu_2_chi, sigma_2_chi, lamb_chi_1, amax
        )

    return (
        _two_gaussian_mixture(
            data['a_1'], mu_1_chi, sigma_1_chi, mu_2_chi, sigma_2_chi, lamb_chi_1, amax
        )
        + _two_gaussian_mixture(
            data['a_2'], mu_1_chi, sigma_1_chi, mu_2_chi, sigma_2_chi, lamb_chi_2, amax
        )
    )


def two_gaussian_spin_fms(data, mu_1_chi, mu_2_chi, sigma_1_chi, sigma_2_chi,
                          lamb_chi_1, lamb_chi_2, amax=1.,
                          NS_amax=NS_SPIN_MAXIMUM, NS_mmax=NS_MASS_MAXIMUM):
    """
    Spin magnitude distribution: mixture of two truncated Gaussians, with a
    neutron-star spin cap.

    :func:`two_gaussian_spin` with the additional requirement that a component
    lighter than `NS_mmax` cannot spin faster than `NS_amax`. This is the GWTC-6
    full-mass-spectrum component-spin model, i.e. the log-space equivalent of
    ``gwtc6_population_models.spin.spin_magnitude_two_gaussians_CBC``.

    Parameters
    ----------
    data : dict
        Must contain 'a_1', 'a_2', and enough mass information to identify
        neutron stars: 'mass_1' or 'log_mass_1', plus one of 'mass_2',
        'log_mass_2' or 'mass_ratio'.
    mu_1_chi : float
        Mean of the first (low-spin) Gaussian component.
    mu_2_chi : float
        Mean of the second (high-spin) Gaussian component.
    sigma_1_chi : float
        Std. deviation of the first Gaussian component.
    sigma_2_chi : float
        Std. deviation of the second Gaussian component.
    lamb_chi_1 : float
        Weight on the first Gaussian for the primary spin.
    lamb_chi_2 : float
        Weight on the first Gaussian for the secondary spin.
    amax : float, optional
        Maximum spin magnitude for black holes (default 1).
    NS_amax : float, optional
        Maximum spin magnitude for neutron stars (default 0.4).
    NS_mmax : float, optional
        Maximum mass for neutron stars (default 2.5).

    Returns
    -------
    jnp.ndarray
        Log-probability density of the spin magnitude distribution.
    """
    masses = _component_masses(data)

    total_prob = jnp.zeros_like(data['a_1'])
    for mass, lamb, key in zip(masses, (lamb_chi_1, lamb_chi_2), ('a_1', 'a_2')):
        def mixture(high, a=data[key], lamb=lamb):
            return _two_gaussian_mixture(
                a, mu_1_chi, sigma_1_chi, mu_2_chi, sigma_2_chi, lamb, high
            )
        total_prob += jnp.where(mass < NS_mmax, mixture(NS_amax), mixture(amax))

    return total_prob


def tilt_iid(data, mu, sig, zeta):
    """
    Assumes the tilt distribution is independent and identically
    distributed across components, using the isotropic + gaussian model

    Parameters
    ----------
    data : dict
        Must contain 'cos_tilt_1' and 'cos_tilt_2'.
    sig : float
        Standard deviation of the truncated Gaussian (mean fixed to 1).
    zeta : float
        Mixture fraction for the field (truncated Gaussian) component.

    Returns
    -------
    jnp.ndarray
        Log-probabilities of the tilt distribution.
    """
    ln_zeta = jnp.log(zeta)
    ln_1mzeta = jnp.log(1 - zeta)

    if ('cos_tilt' in data or 't' in data) and ('cos_tilt_1' not in data and 'cos_tilt_2' not in data):
        # just return the marginal, this is used for saving marginals in popsummary
        if 't' in data:
            costilt = data['t']
        else:
            costilt = data['cos_tilt']
        pfield = trunc_gaussian(costilt, mu, sig, -1, 1)
        pisotropic = jnp.log(jnp.ones_like(costilt) / 2)

        return jnp.logaddexp(ln_zeta + pfield, ln_1mzeta + pisotropic)

    pfield1 = trunc_gaussian(data['cos_tilt_1'], mu, sig, -1, 1)
    pfield2 = trunc_gaussian(data['cos_tilt_2'], mu, sig, -1, 1)

    pisotropic = jnp.log(jnp.ones_like(data['cos_tilt_1']) / 2)

    p1 = jnp.logaddexp(ln_zeta + pfield1, ln_1mzeta + pisotropic)
    p2 = jnp.logaddexp(ln_zeta + pfield2, ln_1mzeta + pisotropic)
    return p1 + p2
