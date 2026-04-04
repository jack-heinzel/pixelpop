import wcosmo
#import unxt # SAL: this needs fixing. Why doesn't it compile?
from jax import jit#, lax
from jax.nn import log_sigmoid
from numpyro import distributions as dist
import jax.numpy as jnp
import jax.scipy.special as scs
import numpy as np
from functools import partial

Planck15_LAL = wcosmo.FlatLambdaCDM(H0=67.90, Om0=0.3065, name="Planck15_LAL")
COSMO = Planck15_LAL
INF = 1e10 # avoid actual jnp.inf, otherwise we get nan gradients

def log_expit(x):
    """
    Numerically stable implementation of log(sigmoid(x)).

    This avoids overflow/underflow by applying a branch split:
    - For x < 0:  x - log1p(exp(x))
    - For x >= 0: -log1p(exp(-x))

    Equivalent to `scipy.special.log_expit`, but implemented with
    JAX-safe `where` to prevent NaN gradients.

    Parameters
    ----------
    x : float or jnp.ndarray
        Input value(s).

    Returns
    -------
    jnp.ndarray
        log(sigmoid(x)) evaluated elementwise.
    """
    condition = x < 0
    posx_valid = jnp.where(condition, 0, x) # in forward differentiation, gradient is 0 for condition, 1 where false
    negx_valid = jnp.where(condition, x, 0) # in forward differentiation, gradient is 0 for condition, 1 where false
    
    return jnp.where(condition, negx_valid-jnp.log1p(jnp.exp(negx_valid)), -jnp.log1p(jnp.exp(-posx_valid)))

def m_smoother(m1s, minimum, delta, buffer=1e-3):
    """
    Apply a smoothing function at the minimum mass cutoff.

    Implements the standard smoothing of a power-law at the low-mass
    edge, following Eq. (B5) of arXiv:2111.03634. Ensures continuity
    across [m_min, m_min + delta].

    Parameters
    ----------
    m1s : jnp.ndarray
        Primary mass values.
    minimum : float
        Minimum allowed mass.
    delta : float
        Width of smoothing region.
    buffer : float, optional
        Small offset to avoid division-by-zero.

    Returns
    -------
    jnp.ndarray
        Log-smoothing factor applied to the mass distribution.
    """
    m_prime = jnp.clip(m1s - minimum, buffer, delta-buffer)
    return log_expit(-delta/m_prime - delta/(m_prime - delta))

def powerlaw(data, slope, minimum, maximum):
    """
    Compute the log-PDF of a truncated power-law distribution.

    Parameters
    ----------
    data : jnp.ndarray
        Evaluation points.
    slope : float
        Power-law exponent.
    minimum : float
        Lower bound of support.
    maximum : float
        Upper bound of support.

    Returns
    -------
    jnp.ndarray
        Log-probability density evaluated at `data`.
        Returns -INF outside [minimum, maximum].
    """
    norm = jnp.where(
        jnp.isclose(slope, -1), 
        jnp.log(jnp.log(maximum / minimum)),
        -jnp.log(jnp.abs(slope + 1)) + jnp.log(jnp.abs(maximum**(slope+1) - minimum**(slope+1)))
    )
    window = jnp.logical_and(data >= minimum, data <= maximum)
    p = jnp.where(window, slope*jnp.log(data), -INF*jnp.ones_like(data))
    return p - norm

def gaussian(data, mean, sig):
    """
    Compute the log-PDF of a Gaussian distribution.

    Parameters
    ----------
    data : jnp.ndarray
        Evaluation points.
    mean : float
        Gaussian mean.
    sig : float
        Standard deviation.

    Returns
    -------
    jnp.ndarray
        Log-probability density evaluated at `data`.
    """
    px = -(data - mean)**2 / 2 / sig**2
    norm = 0.5*jnp.log(2*jnp.pi*sig**2)
    return px - norm

# Custom mixture model
def Peak_PrimaryMass(data, mpp, sigpp):
    isLogMass = True
    if isinstance(data, dict):
        try:
            m1 = jnp.exp(data['log_mass_1'])
        except KeyError:
            isLogMass = False
            m1 = data['mass_1']
    else:
        m1 = data
    pm1 = gaussian(m1, mpp, sigpp)
    if isLogMass: # include jacobian
        pm1 = pm1 + data['log_mass_1']
    return pm1

def TwoPeaks_PrimaryMass(data, mpp1, sigpp1, mpp2, sigpp2, lam):
    isLogMass = True
    if isinstance(data, dict):
        try:
            m1 = jnp.exp(data['log_mass_1'])
        except KeyError:
            isLogMass = False
            m1 = data['mass_1']
    else:
        m1 = data
    pm1 = gaussian(m1, mpp1, sigpp1)
    pm2 = gaussian(m1, mpp2, sigpp2)
    pm = jnp.logaddexp(pm1 + jnp.log(1-lam), pm2 + jnp.log(lam))
    
    if isLogMass: # include jacobian
        pm = pm + data['log_mass_1']
    return pm

def PowerlawPlusPeak_PrimaryMass(data, alpha, minimum, maximum, delta_m, mpp, sigpp, lam):
    """
    Power-law + Gaussian-peak model for primary BH masses.

    This is the "PP" model used in GWTC catalogs, consisting of:
    - A smoothed power-law at low masses.
    - A Gaussian peak centered at `mpp` with width `sigpp`.
    - A mixture fraction `lam` between the two components.
    - Normalization via simple Riemann integration.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Event data containing 'log_mass_1' or 'mass_1'.
    alpha : float
        Power-law slope (negative exponent).
    minimum : float
        Minimum mass cutoff.
    maximum : float
        Maximum mass cutoff.
    delta_m : float
        Smoothing width at the low-mass edge.
    mpp : float
        Mean of Gaussian peak.
    sigpp : float
        Std. dev. of Gaussian peak.
    lam : float
        Mixture fraction of Gaussian component.

    Returns
    -------
    jnp.ndarray
        Log-probability density for primary mass.
    """
    slope = -alpha
    isLogMass = True
    if isinstance(data, dict):
        try:
            m1 = jnp.exp(data['log_mass_1'])
        except KeyError:
            isLogMass = False
            m1 = data['mass_1']
    else:
        m1 = data
        isLogMass = False
    power_law = powerlaw(m1, slope, minimum, maximum)
    smoothed_pl = power_law + m_smoother(m1, minimum, delta_m)
    peak = gaussian(m1, mpp, sigpp)
    pm1 = jnp.logaddexp(smoothed_pl + jnp.log(1-lam), peak + jnp.log(lam))
    
    m1s_test = jnp.linspace(2.0, 200., 2000)
    dm1 = m1s_test[1] - m1s_test[0]
    power_law_test = powerlaw(m1s_test, slope, minimum, maximum)
    smoothed_pl_test = power_law_test + m_smoother(m1s_test, minimum, delta_m)
    peak_test = gaussian(m1s_test, mpp, sigpp)
    smoothed_pl_test = jnp.logaddexp(smoothed_pl_test + jnp.log(1-lam), peak_test + jnp.log(lam))
    
    pm1 -= scs.logsumexp(smoothed_pl_test) + jnp.log(dm1) # simple Riemann rule
    if isLogMass: # include jacobian
        pm1 = pm1 + data['log_mass_1']
    return pm1


def BrokenPowerLaw(data, slope_1, slope_2, xmin, xmax, break_fraction):
    """
    Broken power-law distribution with a single spectral break.

    Defines a continuous piecewise power-law across [xmin, xmax] with
    slopes `slope_1` (below the break) and `slope_2` (above the break).
    The break location is determined by `break_fraction` of the interval.

    Parameters
    ----------
    data : jnp.ndarray
        Evaluation points.
    slope_1 : float
        Power-law slope below the break.
    slope_2 : float
        Power-law slope above the break.
    xmin : float
        Lower support bound.
    xmax : float
        Upper support bound.
    break_fraction : float
        Fractional location of the break within [xmin, xmax].

    Returns
    -------
    jnp.ndarray
        Log-probability density of the broken power-law distribution.
    """
    m_break = xmin + break_fraction * (xmax - xmin)
    correction = powerlaw(m_break, slope_2, m_break, xmax) - powerlaw(
        m_break, slope_1, xmin, m_break
    )
    low_part = powerlaw(data, slope_1, xmin, m_break)
    high_part = powerlaw(data, slope_2, m_break, xmax)
    
    # this might be nan gradient?
    prob = jnp.where(data < m_break, low_part + correction, high_part)

    return prob + log_sigmoid(-correction) # - log(1+exp(correction))

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

def chieff_gaussian(data, mean, sig):
    """
    Effective spin distribution: Gaussian in chi_eff.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict containing key 'chi_eff', or direct array of chi_eff values.
    mean : float
        Mean of the Gaussian.
    sig : float
        Standard deviation of the Gaussian.

    Returns
    -------
    jnp.ndarray
        Log-probability density under the Gaussian distribution.
    """
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    return gaussian(x, mean, sig)

def chip_gaussian(data, mean, sig):
    """
    Effective precessing spin distribution: Gaussian in chi_p.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict containing key 'chi_p', or direct array of chi_p values.
    mean : float
        Mean of the Gaussian.
    sig : float
        Standard deviation of the Gaussian.

    Returns
    -------
    jnp.ndarray
        Log-probability density under the Gaussian distribution.
    """
    if isinstance(data, dict):
        x = data['chi_p']
    else:
        x = data
    return trunc_gaussian(x, mean, sig, 0, 1)


def trunc_gaussian(data, mean, sig, lower, upper):
    """
    Truncated Gaussian distribution. Numerically stable implementation adapted from
    https://github.com/ColmTalbot/gwpopulation/blob/6e60056be9ae809515eb4576e1ab581c5607a49c/gwpopulation/utils.py#L133-L183

    Parameters
    ----------
    data : jnp.ndarray
        Evaluation points.
    mean : float
        Mean of the Gaussian.
    sig : float
        Standard deviation of the Gaussian.
    lower : float
        Lower truncation bound.
    upper : float
        Upper truncation bound.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the truncated Gaussian,
        with −INF outside [lower, upper].
    """
    
    def logsubexp(log_p, log_q):
        return log_p + jnp.log(1 - jnp.exp(log_q - log_p))

    up = (upper - mean) / sig
    lo = (lower - mean) / sig

    px = -(data - mean)**2 / 2 / sig**2 - np.log(2.0 * np.pi) / 2.0 - jnp.log(sig)

    # cf https://github.com/scipy/scipy/blob/v1.15.1/scipy/stats/_continuous_distns.py#L10189
    log_norm = jnp.select(
        [up <= 0, lo > 0, up > 0],
        [
            logsubexp(scs.log_ndtr(up), scs.log_ndtr(lo)),
            logsubexp(scs.log_ndtr(-lo), scs.log_ndtr(-up)),
            jnp.log1p(-scs.ndtr(lo) - scs.ndtr(-up)),
        ],
        jnp.nan,
    )
    px -= log_norm
    in_support = jnp.logical_and(data < upper, data > lower)
    return jnp.where(in_support, px, -jnp.inf*jnp.ones_like(data))

def lognormal(data, mean, sig):
    """
    Log-normal distribution.

    Parameters
    ----------
    data : jnp.ndarray
        Evaluation points (must be > 0).
    mean : float
        location parameter of lognormal (mean  of ln(X) if X~LogNormal)
    sig : float
        width parameter of lognormal (standard deviation of ln(X) if X~LogNormal)

    Returns
    -------
    jnp.ndarray
        Log-probability density of the log-normal distribution.
    """
    px = -(jnp.log(data) - mean)**2 / 2 / sig**2
    denom = jnp.log(data*sig*jnp.sqrt(2*jnp.pi))
    return px - denom

# TODO: base Redshift function that takes in Psi evolution and returns log density
def PowerlawRedshift(data, lamb, max_z=1.9, normalize=True, return_normalization=False):
    """
    Redshift distribution model: power law in (1+z) weighted by comoving volume.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict containing key 'redshift', or direct array of redshifts.
    lamb : float
        Power-law index on (1+z).
    max_z : float, optional
        Maximum redshift cutoff (default 1.9).
    normalize : bool, optional
        If True, normalize the distribution (default True).
    return_normalization : bool, optional
        If True, return only the log-normalization constant.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the redshift distribution.
        If `return_normalization=True`, returns the log-normalization only.
    """
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    zs_fixed = jnp.linspace(1e-5, max_z, 1000)
    dvs = COSMO.differential_comoving_volume(zs_fixed)
    if isinstance(dvs, unxt.quantity.Quantity):
        # TODO: preferably would use unxt.unit values...
        dvs = 4*jnp.pi * 1e-9 * dvs.value
    else:
        dvs = 4*jnp.pi * 1e-9 * dvs
    fixed_ln_dvc_dz = jnp.log(dvs)

    if normalize:
        dz = zs_fixed[1] - zs_fixed[0]
        test_ln_p = fixed_ln_dvc_dz + (lamb - 1) * jnp.log(1. + zs_fixed)
        ln_norm = scs.logsumexp(test_ln_p) + jnp.log(dz)
        if return_normalization:
            return ln_norm
    else:
        ln_norm = 0.
    ln_dvc_dz = jnp.interp(z, zs_fixed, fixed_ln_dvc_dz)
    ln_p = ln_dvc_dz + (lamb - 1) * jnp.log(1. + z)
    ln_p -= ln_norm
        
    window = jnp.logical_and(z >= 0., z <= max_z)
    p = jnp.where(window, ln_p, -INF*jnp.ones_like(z))
    return p

def PowerlawRedshiftPsi(data, lamb, max_z=1.9):
    """
    Power-law redshift distribution: proportional to (1+z)^lamb.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict containing key 'redshift', or direct array of redshifts.
    lamb : float
        Power-law index on (1+z).
    max_z : float, optional
        Maximum redshift cutoff (default 1.9).

    Returns
    -------
    jnp.ndarray
        Log-probability density, with −INF outside [0, max_z].
    """
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    ln_p = lamb * jnp.log(1. + z)

    window = jnp.logical_and(z >= 0., z <= max_z)
    p = jnp.where(window, ln_p, -INF*jnp.ones_like(z))
    return p

def _log_expm1_stable(a):
    """Compute log(expm1(a)) stably for real a (a can be small/large)."""
    return jnp.where(a > 0,
                     a + jnp.log1p(-jnp.exp(-a)),
                     jnp.log(jnp.expm1(a)))

def NormalizedPowerlawRedshiftPsi(data, lamb, max_z=1.9):
    """Normalized (1+z)^lambda density over [min_z, max_z]."""
    tol = 1e-8
    z = data["redshift"] if isinstance(data, dict) else data

    # unnormalized log shape
    ln_unnorm = lamb * jnp.log1p(z)

    # log normalization constant logZ
    a = (lamb + 1.0) * jnp.log1p(max_z)

    logZ_general = _log_expm1_stable(a) - jnp.log(lamb + 1.0)
    logZ_lm1 = jnp.log(jnp.log1p(max_z))  # log( log(1+max_z) )

    logZ = jnp.where(jnp.abs(lamb + 1.0) < tol, logZ_lm1, logZ_general)

    # cutoff outside [0, max_z]
    window = (z >= 0.0) & (z <= max_z)
    ln_p = ln_unnorm - logZ
    return jnp.where(window, ln_p, -INF * jnp.ones_like(z))

def MadauDickinsonRedshift(data, gamma, kappa, z_peak, z_max=1.9, normalize=True, return_normalization=False):
    """
    Madau–Dickinson star-formation rate redshift distribution.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict containing key 'redshift', or direct array of redshifts.
    gamma : float
        Low-redshift power-law index.
    kappa : float
        High-redshift suppression exponent.
    z_peak : float
        Characteristic peak redshift.
    z_max : float, optional
        Maximum redshift cutoff (default 1.9).
    normalize : bool, optional
        If True, normalize the distribution (default True).
    return_normalization : bool, optional
        If True, return only the log-normalization constant.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the redshift distribution.
        If `return_normalization=True`, returns the log-normalization only.
    """
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    zs_fixed = np.linspace(1e-5, z_max, 1000)
    fixed_ln_dvc_dz = jnp.log(
        4*jnp.pi*COSMO.differential_comoving_volume(zs_fixed).to(unxt.Gpc**3 / unxt.sr).value
        )
    if normalize:
        dz = zs_fixed[1] - zs_fixed[0]
        test_ln_p = fixed_ln_dvc_dz + (gamma - 1)* jnp.log(1. + zs_fixed) - jnp.log(1 + ((1 + zs_fixed)/(1 + z_peak))**kappa)
        ln_norm = scs.logsumexp(test_ln_p) + jnp.log(dz)
        if return_normalization:
            return ln_norm
    else:
        ln_norm = 0.
    ln_dvc_dz = jnp.interp(z, zs_fixed, fixed_ln_dvc_dz)
    ln_p = ln_dvc_dz + (gamma - 1)* jnp.log(1. + z) - jnp.log(1 + ((1 + z)/(1 + z_peak))**kappa)
    ln_p -= ln_norm

    window = jnp.logical_and(z >= 0., z <= z_max)
    p = jnp.where(window, ln_p, -INF*jnp.ones_like(z))
    return p

def PowerlawPlusPeak_MassRatio(data, slope, minimum, delta_m):
    """
    Mass-ratio distribution: smoothed power law with minimum mass cut.

    Parameters
    ----------
    data : dict
        Must contain 'mass_ratio' and either 'mass_1' or 'log_mass_1'.
    slope : float
        Power-law slope on the mass ratio q.
    minimum : float
        Global minimum BH mass.
    delta_m : float
        Mass smoothing scale at the minimum cutoff.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the smoothed mass-ratio distribution.
    """

    try:
        m1 = jnp.exp(data['log_mass_1'])
    except KeyError:
        m1 = data['mass_1']
    q = data['mass_ratio']

    power_law = powerlaw(q, slope, minimum/m1, jnp.ones_like(m1))
    smoothed_pl = power_law + m_smoother(q*m1, minimum, delta_m)

    m1s_test = jnp.exp(jnp.linspace(jnp.log(2.), jnp.log(100.), 500))
    m2s_test = jnp.linspace(1.99*jnp.ones_like(m1s_test), m1s_test, 10000)
    qs_test = m2s_test / jnp.expand_dims(m1s_test, axis=0)
    dq = qs_test[1] - qs_test[0]
    power_law_test = powerlaw(qs_test, slope, 0.02, 1.) # fiducial lower bound of 0.02 
    smoothed_pl_test = power_law_test + m_smoother(m2s_test, minimum, delta_m)
    
    norm = scs.logsumexp(smoothed_pl_test, axis=0) + jnp.log(dq) # simple Riemann rule
    # norms = jnp.interp(m1, m1s_test, norm)
    norms = norm[jnp.digitize(m1, m1s_test)] # take the point to the right of each m1, so
    # that the normalization is always SMALLER than the true value, so that 
    # correct normalization from fiducial lower bound
    norms += jnp.log(jnp.abs(1 - 0.02**(slope+1))) - jnp.log(jnp.abs(1 - (minimum/m1)**(slope+1)))
    return smoothed_pl - norms

def Powerlaw_MassRatio(data, slope, minimum):
    """
    Simple power-law mass-ratio distribution with a global minimum mass.

    Parameters
    ----------
    data : dict
        Must contain 'mass_ratio' and either 'mass_1' or 'log_mass_1'.
    slope : float
        Power-law slope on q.
    minimum : float
        Global minimum BH mass.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the mass-ratio distribution.
    """
    try:
        m1 = jnp.exp(data['log_mass_1'])
    except KeyError:
        m1 = data['mass_1']
    q = data['mass_ratio']

    power_law = powerlaw(q, slope, minimum/m1, jnp.ones_like(m1))
    return power_law

def SimplePowerlaw_MassRatio(data, slope, qmin):
    """
    Simple power-law mass-ratio distribution without a global minimum mass.

    Parameters
    ----------
    data : dict
        Must contain 'mass_ratio'.
    slope : float
        Power-law slope on q.
    qmin : float
        Minimum mass ratio allowed.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the mass-ratio distribution.
    """
    q = data['mass_ratio']

    power_law = powerlaw(q, slope, qmin, 1.)
    return power_law


def PowerlawPlusPeak(data, alpha, beta, mmin, mmax, delta_m, mpp, sigpp, lam):
    """
    Joint primary-mass and mass-ratio distribution: power law plus Gaussian peak.

    Parameters
    ----------
    data : dict
        Must contain 'mass_1' or 'log_mass_1', and 'mass_ratio'.
    alpha : float
        Power-law slope for primary mass.
    beta : float
        Power-law slope for mass ratio.
    mmin, mmax : float
        Minimum and maximum primary masses.
    delta_m : float
        Smoothing scale at lower mass cutoff.
    mpp : float
        Peak mass location.
    sigpp : float
        Standard deviation of the peak Gaussian.
    lam : float
        Fraction in the Gaussian peak.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the joint distribution.
    """

    pm1 = PowerlawPlusPeak_PrimaryMass(data, alpha, mmin, mmax, delta_m, mpp, sigpp, lam)
    pq = PowerlawPlusPeak_MassRatio(data, beta, mmin, delta_m)

    return pm1 + pq

def smooth(x, cutoff, width):
    """
    Smooth cutoff function with continuous derivative.

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points.
    cutoff : float
        Cutoff location.
    width : float
        Width of smoothing region.

    Returns
    -------
    jnp.ndarray
        Smooth step function, transitioning quadratically at cutoff.
    """
    return jnp.where(x<cutoff, 0., -((x-cutoff)/width)**2)

def mu_var_to_alpha_beta(mu, var):
    """
    Convert mean and variance to Beta distribution parameters.

    Parameters
    ----------
    mu : float
        Mean of the Beta distribution.
    var : float
        Variance of the Beta distribution.

    Returns
    -------
    alpha : float
        Beta shape parameter alpha.
    beta : float
        Beta shape parameter beta.
    """
    nu = (mu*(1-mu)/var) - 1
    alpha = mu * nu
    beta = (1-mu) * nu
    return alpha, beta

def beta_spin(spin_mag, alpha, beta):
    """
    Beta distribution for spin magnitudes.

    Parameters
    ----------
    spin_mag : jnp.ndarray
        Spin magnitudes in [0, 1].
    alpha : float
        Beta distribution parameter alpha.
    beta : float
        Beta distribution parameter beta.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the Beta distribution.
    """
    ln_a = jnp.log(spin_mag)
    ln_1ma = jnp.log(1. - spin_mag)
    ln_p = (alpha - 1) * ln_a + (beta - 1) * ln_1ma

    norm = scs.gammaln(alpha) + scs.gammaln(beta) - scs.gammaln(alpha + beta)
    return ln_p - norm

def beta_spin_mv(spin_mag, mu, var):
    """
    Beta distribution for spin magnitudes.

    Parameters
    ----------
    spin_mag : jnp.ndarray
        Spin magnitudes in [0, 1].
    mu : float
        Beta distribution mean.
    var : float
        Beta distribution variance.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the Beta distribution.
    """
    alpha, beta = mu_var_to_alpha_beta(mu, var)
    return beta_spin(spin_mag, alpha, beta)

def iid_beta_spin(data, mu, var):
    """
    Beta distribution for spin magnitudes.

    Parameters
    ----------
    data : dict
        Must contain 'a_1' and 'a_2'
    mu : float
        Beta distribution mean.
    var : float
        Beta distribution variance.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the Beta distribution.
    """
    alpha, beta = mu_var_to_alpha_beta(mu, var)
    return beta_spin(data['a_1'], alpha, beta) + beta_spin(data['a_2'], alpha, beta)

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
    return trunc_gaussian(data['a_1'], mu, sig, 0, 1) + trunc_gaussian(data['a_2'], mu, sig, 0, 1)

def iid_normal_spin_fms(data, mu, var, NS_amax=0.4, NS_mmax=2.5):
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
    keys = data.keys()
    if 'mass_1' in keys:
        m1 = data['mass_1']
    elif 'log_mass_1' in keys:
        m1 = jnp.exp(data['log_mass_1'])
    if 'mass_ratio' in keys:
        m2 = m1 * data['mass_ratio']
    elif 'log_mass_2' in keys:
        m2 = jnp.exp(data['log_mass_2'])
    regions = {'mass_1': m1, 'mass_2': m2}
    for ii in [1,2]:
        probs = jnp.where(
            regions[f'mass_{ii}'] < NS_mmax, 
            trunc_gaussian(data[f'a_{ii}'], mu, sig, 0, NS_amax), 
            trunc_gaussian(data[f'a_{ii}'], mu, sig, 0, 1)
            )
        total_prob += probs
    
    return total_prob

def tilt_model(data, mu, sig, zeta):
    """
    Tilt distribution model allowing a free mean tilt parameter.

    Models the joint distribution of the cosine tilts of both black holes
    (`cos_tilt_1`, `cos_tilt_2`) as either:
      - a truncated Gaussian centered at `mu` with width `sig`, with probability `zeta`, or
      - an isotropic distribution, with probability `1 - zeta`.

    Parameters
    ----------
    data : dict
        Must contain 'cos_tilt_1' and 'cos_tilt_2'.
    mu : float
        Mean of the truncated Gaussian.
    sig : float
        Standard deviation of the truncated Gaussian.
    zeta : float
        Mixture fraction for the field (truncated Gaussian) component.

    Returns
    -------
    jnp.ndarray
        Log-probabilities of the tilt distribution.
    """
    pfield1 = trunc_gaussian(data['cos_tilt_1'], mu, sig, -1, 1)
    pfield2 = trunc_gaussian(data['cos_tilt_2'], mu, sig, -1, 1)

    pisotropic = jnp.log(jnp.ones_like(data['cos_tilt_1']) / 4)
    pfield = pfield1 + pfield2

    ln_zeta = jnp.log(zeta)
    ln_1mzeta = jnp.log(1 - zeta)

    return jnp.logaddexp(ln_zeta + pfield, ln_1mzeta + pisotropic)


def tilt_default(data, sig, zeta):
    """
    Default tilt distribution model.

    Assumes the tilt distribution is not independent across components:
    either both tilts are isotropic or both follow a truncated Gaussian
    centered at `mu=1`.

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
    return tilt_model(data, 1., sig, zeta)


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
    pfield1 = trunc_gaussian(data['cos_tilt_1'], mu, sig, -1, 1)
    pfield2 = trunc_gaussian(data['cos_tilt_2'], mu, sig, -1, 1)

    pisotropic = jnp.log(jnp.ones_like(data['cos_tilt_1']) / 2)
    
    ln_zeta = jnp.log(zeta)
    ln_1mzeta = jnp.log(1 - zeta)

    p1 = jnp.logaddexp(ln_zeta + pfield1, ln_1mzeta + pisotropic)
    p2 = jnp.logaddexp(ln_zeta + pfield2, ln_1mzeta + pisotropic)
    return p1 + p2
    

def spin_iid(data, mu, var, mu_tilt, sig_tilt, zeta):
    return iid_normal_spin(data, mu, var) + tilt_iid(data, mu_tilt, sig_tilt, zeta)

def gwtc3_spin_default(data, mu, var, sig_tilt, zeta):
    return iid_beta_spin(data, mu, var) + tilt_default(data, sig_tilt, zeta)

def spin_default(data, mu, var, sig_tilt, zeta):
    return iid_beta_spin(data, mu, var) + tilt_default(data, sig_tilt, zeta)

def logpdf_chieff_correlated_linear(
    data,
    mu_chieff_0,
    mu_chieff_1,
    ln_sigma_chieff_0,
    ln_sigma_chieff_1,
    x0,
    parameter,   # string: "mass_ratio" or "redshift" 
    amax=1.0,
    min_sigma=1e-6,
):
    """
    Log p(chi_eff | x) where chi_eff is TruncatedNormal(mu(x), sigma(x), [-amax, amax]),
    and mu(x), ln sigma(x) are linear in (x - x0).

    Returns: array of log-prob with the same broadcasted shape as chi_eff and x.
    """
    if isinstance(data, dict):
        chi_eff = data['chi_eff']
    else:
        chi_eff = data

    x = data[parameter]

    # Linear mean and log-sigma
    mu = mu_chieff_0 + mu_chieff_1 * (x - x0)
    ln_sigma = ln_sigma_chieff_0 + ln_sigma_chieff_1 * (x - x0)

    # sigma must be > 0; clip to avoid numerical issues
    sigma = jnp.exp(ln_sigma)
    sigma = jnp.maximum(sigma, min_sigma)

    # Truncated normal on [-amax, amax]
    logp = trunc_gaussian(chi_eff, mu, sigma, lower=-amax, upper=amax)

    # Match gwpopulation's nan_to_num(posinf=0) behavior in LOGSPACE:
    # - any nan/inf should become -inf (i.e., probability 0)
    logp = jnp.where(jnp.isfinite(logp), logp, -INF)
    return logp

def trunc_gaussian_lims(data, mean, sig, lower, upper):
    px = -(data - mean)**2 / 2 / sig**2
    width = 0.001 #Hardcoding it for now
    taper_l = jnp.where(data > lower, 0, ((data-lower)/width))
    taper_r = jnp.where(data < upper, 0, -((data-upper)/width))
    px = px + taper_l + taper_r
    up = (upper - mean) / sig / jnp.sqrt(2)
    lo = (lower - mean) / sig / jnp.sqrt(2)
    trunc = 0.5*(scs.erf(up) - scs.erf(lo))
    norm = 0.5*jnp.log(2*jnp.pi*sig**2) + jnp.log(trunc)
    return px - norm

def log_tukey_norm(x, tx0, tr, tk, eps=1e-10, normalize=True):
    # Define parameters to add/subtract
    x0 = jnp.where(-1 > tx0 - tk, -1, tx0 - tk)
    x1 = tx0 - tk * (1 - tr)
    x2 = tx0 + tk * (1 - tr)
    x3 = jnp.where(1 < tx0 + tk, 1, tx0 + tk)
    
    # Adding an epsilon to avoid numerical errors
    ln_t01 = jnp.log((1 + jnp.cos((x - x1) * jnp.pi / tk / tr)) / 2 + eps)
    ln_t23 = jnp.log((1 + jnp.cos((x - x2) * jnp.pi / tk / tr)) / 2 + eps)
    
    # Applying the windowing
    ln_t = jnp.ones(x.shape)*(-100) # All values outside near -inf
    ln_t = jnp.where((x0 <= x) * (x < x1), ln_t01, ln_t)
    ln_t = jnp.where((x1 <= x) * (x < x2), 0, ln_t)
    ln_t = jnp.where((x2 <= x) * (x < x3), ln_t23, ln_t)
    
    
    # Calculate the normalization
    if normalize:
        xs_fixed = jnp.linspace(-1, 1, 1000)
        dx = xs_fixed[1] - xs_fixed[0]
        ln_t01_n = jnp.log((1 + jnp.cos((xs_fixed - x1) * jnp.pi / tk / tr)) / 2 + eps)
        ln_t23_n = jnp.log((1 + jnp.cos((xs_fixed - x2) * jnp.pi / tk / tr)) / 2 + eps)
    
        t_n = jnp.ones(xs_fixed.shape)*(-100) # All values outside near -inf
        t_n = jnp.where((x0 <= xs_fixed) * (xs_fixed < x1), ln_t01_n, t_n)
        t_n = jnp.where((x1 <= xs_fixed) * (xs_fixed < x2), 0, t_n)
        t_n = jnp.where((x2 <= xs_fixed) * (xs_fixed < x3), ln_t23_n, t_n) 
    
        t_norm = scs.logsumexp(t_n) + jnp.log(dx)
        ln_t -= t_norm
        
    window = jnp.logical_and(x >= -1, x <= 1)
    t = jnp.where(window, ln_t, -100.*jnp.ones_like(x))
    
    return t

def chieff_gaussian_tukey_original_parametrization(data, mean, sig, tx0, tr, tk, lamb_x):
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    gaussian = trunc_gaussian_lims(x, mean, sig, lower=-1, upper=1)
    tukey = log_tukey_norm(x, tx0, tr, tk)
    model = jnp.logaddexp(jnp.log(lamb_x) + gaussian, jnp.log(1 - lamb_x) + tukey)
    return model

def chieff_skewed_tukey(data, mean, sig, skew, tx0, tr, tk, lamb_x):
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    gaussian = trunc_gaussian_lims(x, mean, sig, lower=-1, upper=1)
    skewed = chieff_skew_gaussian(x, mean, sig, skew)
    tukey = log_tukey_norm(x, tx0, tr, tk)
    model = jnp.logaddexp(jnp.log(lamb_x) + gaussian, jnp.log(1 - lamb_x) + tukey)
    return model


def logpdf_1g1g_chieff(data, mu_x, sig_x, chieff_min=-1.0, chieff_max=1.0):
    
    chi_eff = data['chi_eff']
    logp_eff = trunc_gaussian(chi_eff, mu_x, sig_x, chieff_min, chieff_max)
    return logp_eff 

def logpdf_1g1g_chip(data, mu_xp, sigma_xp, chip_min=0.0, chip_max=1.0):
    chi_p = data['chi_p']
    logp_p   = trunc_gaussian(chi_p, mu_xp, sigma_xp, chip_min, chip_max)
    return logp_p 



@partial(jit, static_argnames=['rate_likelihood','return_likelihood_info'])
def hierarchical_likelihood(event_weights, denominator_weights, total_injections, live_time=1, rate_likelihood=False, return_likelihood_info=True):
    """
    Hierarchical Bayesian likelihood for population inference.

    Parameters
    ----------
    event_weights : jnp.ndarray
        Array (n_events, n_samples) of log[p(θ|pop)/pi(θ|PE)] for each event posterior sample.
    denominator_weights : jnp.ndarray
        Array of log[p(θ|pop)/pi(θ|draw)] for injections.
    total_injections : int
        Number of injection samples.
    live_time : float, optional
        Observing time in years (default 1).
    rate_likelihood : bool, optional
        If True, include rate likelihood (default False).
    return_likelihood_info : bool, optional
        If True, return decomposition of likelihood and variances.

    Returns
    -------
    tuple
        If `return_likelihood_info=True`:
            (lnL, var, [pe_lnL, vt_lnL], [pe_var, vt_var])
        else:
            (lnL, var)
    """
    n_events, minimum_length = event_weights.shape
    numerators = scs.logsumexp(event_weights, axis=1) - jnp.log(minimum_length) # means
    denominator = scs.logsumexp(denominator_weights) - jnp.log(total_injections)

    pe_ln_likelihood = jnp.sum(numerators)
    if rate_likelihood:
        vt_ln_likelihood = n_events*jnp.log(live_time) - live_time*jnp.exp(denominator)
    else:
        vt_ln_likelihood = -n_events*denominator

    ln_likelihood = pe_ln_likelihood + vt_ln_likelihood
    
    square_sums = scs.logsumexp(2*event_weights, axis=1) - 2*jnp.log(minimum_length) # square_sums
    square_sum = scs.logsumexp(2*denominator_weights) - 2*jnp.log(total_injections)
    
    pe_ln_likelihood_variance = jnp.sum(jnp.exp(square_sums - 2*numerators) - 1/minimum_length)
    if rate_likelihood:
        vt_ln_likelihood_variance = live_time**2 * (jnp.exp(square_sum) - jnp.exp(2*denominator)/total_injections)
    else:
        vt_ln_likelihood_variance = n_events**2 * (jnp.exp(square_sum - 2*denominator) - 1/total_injections)
    
    ln_likelihood_variance = pe_ln_likelihood_variance + vt_ln_likelihood_variance
    
    if return_likelihood_info:
        ln_likelihoods = [pe_ln_likelihood, vt_ln_likelihood]
        ln_likelihood_variances = [pe_ln_likelihood_variance, vt_ln_likelihood_variance]
        return ln_likelihood, ln_likelihood_variance, ln_likelihoods, ln_likelihood_variances
    else:
        return ln_likelihood, ln_likelihood_variance

def rate_likelihood(event_weights, denominator_weights, total_injections, live_time=1):
    """
    Poisson rate likelihood for hierarchical inference.

    Parameters
    ----------
    event_weights : jnp.ndarray
        Array (n_events, n_samples) of log[p(θ|pop)/pi(θ|PE)] for each event posterior sample.
    denominator_weights : jnp.ndarray
        Array of log[p(θ|pop)/pi(θ|draw)] for injections.
    total_injections : int
        Number of injection samples.
    live_time : float, optional
        Observing time in years (default 1).

    Returns
    -------
    tuple
        (lnL, expected_events, pe_var, vt_var, total_var)
    """
    n_events, minimum_length = event_weights.shape
    numerators = scs.logsumexp(event_weights, axis=1) - jnp.log(minimum_length) # means
    denominator = scs.logsumexp(denominator_weights) - jnp.log(total_injections)

    pe_ln_likelihood = jnp.sum(numerators)

    nexp = live_time*jnp.exp(denominator)
    vt_ln_likelihood = n_events*jnp.log(live_time) - nexp
    ln_likelihood = pe_ln_likelihood + vt_ln_likelihood
    
    square_sums = scs.logsumexp(2*event_weights, axis=1) - 2*jnp.log(minimum_length) # square_sums
    square_sum = scs.logsumexp(2*denominator_weights) - 2*jnp.log(total_injections)
    
    pe_ln_likelihood_variance = jnp.sum(jnp.exp(square_sums - 2*numerators) - 1/minimum_length)
    vt_ln_likelihood_variance = live_time**2 * (jnp.exp(square_sum) - jnp.exp(2*denominator)/total_injections)
    
    ln_likelihood_variance = pe_ln_likelihood_variance + vt_ln_likelihood_variance
    return ln_likelihood, nexp, pe_ln_likelihood_variance, vt_ln_likelihood_variance, ln_likelihood_variance


import wcosmo
import unxt
from jax import jit#, lax
from jax.nn import log_sigmoid
from numpyro import distributions as dist
import jax.numpy as jnp
import jax.scipy.special as scs
import numpy as np
from functools import partial

Planck15_LAL = wcosmo.FlatLambdaCDM(H0=67.90, Om0=0.3065, name="Planck15_LAL")
COSMO = Planck15_LAL
INF = 1e10 # avoid actual jnp.inf, otherwise we get nan gradients

def log_expit(x):
    """
    Numerically stable implementation of log(sigmoid(x)).

    This avoids overflow/underflow by applying a branch split:
    - For x < 0:  x - log1p(exp(x))
    - For x >= 0: -log1p(exp(-x))

    Equivalent to `scipy.special.log_expit`, but implemented with
    JAX-safe `where` to prevent NaN gradients.

    Parameters
    ----------
    x : float or jnp.ndarray
        Input value(s).

    Returns
    -------
    jnp.ndarray
        log(sigmoid(x)) evaluated elementwise.
    """
    condition = x < 0
    posx_valid = jnp.where(condition, 0, x) # in forward differentiation, gradient is 0 for condition, 1 where false
    negx_valid = jnp.where(condition, x, 0) # in forward differentiation, gradient is 0 for condition, 1 where false
    
    return jnp.where(condition, negx_valid-jnp.log1p(jnp.exp(negx_valid)), -jnp.log1p(jnp.exp(-posx_valid)))

def m_smoother(m1s, minimum, delta, buffer=1e-3):
    """
    Apply a smoothing function at the minimum mass cutoff.

    Implements the standard smoothing of a power-law at the low-mass
    edge, following Eq. (B5) of arXiv:2111.03634. Ensures continuity
    across [mmin, mmin + delta].

    Parameters
    ----------
    m1s : jnp.ndarray
        Primary mass values.
    minimum : float
        Minimum allowed mass.
    delta : float
        Width of smoothing region.
    buffer : float, optional
        Small offset to avoid division-by-zero.

    Returns
    -------
    jnp.ndarray
        Log-smoothing factor applied to the mass distribution.
    """
    m_prime = jnp.clip(m1s - minimum, buffer, delta-buffer)
    return log_expit(-delta/m_prime - delta/(m_prime - delta))

def powerlaw(data, slope, minimum, maximum):
    """
    Compute the log-PDF of a truncated power-law distribution.

    Parameters
    ----------
    data : jnp.ndarray
        Evaluation points.
    slope : float
        Power-law exponent.
    minimum : float
        Lower bound of support.
    maximum : float
        Upper bound of support.

    Returns
    -------
    jnp.ndarray
        Log-probability density evaluated at `data`.
        Returns -INF outside [minimum, maximum].
    """
    norm = jnp.where(
        jnp.isclose(slope, -1), 
        jnp.log(jnp.log(maximum / minimum)),
        -jnp.log(jnp.abs(slope + 1)) + jnp.log(jnp.abs(maximum**(slope+1) - minimum**(slope+1)))
    )
    window = jnp.logical_and(data >= minimum, data <= maximum)
    p = jnp.where(window, slope*jnp.log(data), -INF*jnp.ones_like(data))
    return p - norm

def gaussian(data, mean, sig):
    """
    Compute the log-PDF of a Gaussian distribution.

    Parameters
    ----------
    data : jnp.ndarray
        Evaluation points.
    mean : float
        Gaussian mean.
    sig : float
        Standard deviation.

    Returns
    -------
    jnp.ndarray
        Log-probability density evaluated at `data`.
    """
    px = -(data - mean)**2 / 2 / sig**2
    norm = 0.5*jnp.log(2*jnp.pi*sig**2)
    return px - norm

def PowerlawPlusPeak_PrimaryMass(data, alpha, minimum, maximum, delta_m, mpp, sigpp, lam):
    """
    Power-law + Gaussian-peak model for primary BH masses.

    This is the "PP" model used in GWTC catalogs, consisting of:
    - A smoothed power-law at low masses.
    - A Gaussian peak centered at `mpp` with width `sigpp`.
    - A mixture fraction `lam` between the two components.
    - Normalization via simple Riemann integration.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Event data containing 'log_mass_1' or 'mass_1'.
    alpha : float
        Power-law slope (negative exponent).
    minimum : float
        Minimum mass cutoff.
    maximum : float
        Maximum mass cutoff.
    delta_m : float
        Smoothing width at the low-mass edge.
    mpp : float
        Mean of Gaussian peak.
    sigpp : float
        Std. dev. of Gaussian peak.
    lam : float
        Mixture fraction of Gaussian component.

    Returns
    -------
    jnp.ndarray
        Log-probability density for primary mass.
    """
    slope = -alpha
    isLogMass = True
    if isinstance(data, dict):
        try:
            m1 = jnp.exp(data['log_mass_1'])
        except KeyError:
            isLogMass = False
            m1 = data['mass_1']
    else:
        m1 = data
        isLogMass = False
    power_law = powerlaw(m1, slope, minimum, maximum)
    smoothed_pl = power_law + m_smoother(m1, minimum, delta_m)
    peak = gaussian(m1, mpp, sigpp)
    pm1 = jnp.logaddexp(smoothed_pl + jnp.log(1-lam), peak + jnp.log(lam))
    
    m1s_test = jnp.linspace(2.0, 200., 2000)
    dm1 = m1s_test[1] - m1s_test[0]
    power_law_test = powerlaw(m1s_test, slope, minimum, maximum)
    smoothed_pl_test = power_law_test + m_smoother(m1s_test, minimum, delta_m)
    peak_test = gaussian(m1s_test, mpp, sigpp)
    smoothed_pl_test = jnp.logaddexp(smoothed_pl_test + jnp.log(1-lam), peak_test + jnp.log(lam))
    
    pm1 -= scs.logsumexp(smoothed_pl_test) + jnp.log(dm1) # simple Riemann rule
    if isLogMass: # include jacobian
        pm1 = pm1 + data['log_mass_1']
    return pm1


def PlanckWindow_PrimaryMass(data, mmin, delta_m):
    """
    Planck-taper style window acting on log(m1), returned in log-space.

    This is intended to be used as a *multiplicative* window on the merger
    rate in linear space, so the returned quantity is added to other
    log-densities. It does **not** enforce any normalization; it is purely
    a shape modifier.

    Parameters
    ----------
    data : dict or jnp.ndarray
        During inference, this will be the full event/injection dictionary
        containing at least 'log_mass_1' or 'mass_1'. When called from
        ``save_popsummary`` it may instead be a dict with key
        'log_mass_1_window' and a 1D grid.
    mmin : float
        Window edges in log-mass space.
    delta_m : float
        Taper width (0 < delta_m).

    Returns
    -------
    jnp.ndarray
        log w(log_m1), suitable for adding to the population log-density.
    """
    if isinstance(data, dict):
        if 'log_mass_1' in data:
            x = jnp.exp(data['log_mass_1'])
        elif 'mass_1' in data:
            x = data['mass_1']
        else:
            raise KeyError("PlanckWindow_PrimaryMass expects 'log_mass_1', or 'mass_1' in data.")
    else:
        # Assume data is already mass_1.
        x = data

    return m_smoother(x, mmin, delta_m)

def PlanckWindow_PrimaryMassSecondaryMass_TwoMmin(data, mmin_1, delta_m_1, mmin_2, delta_m_2):
    """
    Planck-taper style window acting on m1 and m2, returned in log-space.
    """
    if 'log_mass_1' in data:
        m1 = jnp.exp(data['log_mass_1'])
    elif 'mass_1' in data:
        m1 = data['mass_1']
    else:
        raise KeyError("PlanckWindow_PrimaryMassSecondaryMass_TwoMmin expects 'log_mass_1', or 'mass_1' in data.")
    if 'log_mass_2' in data:
        m2 = jnp.exp(data['log_mass_2'])
    elif 'mass_2' in data:
        m2 = data['mass_2']
    elif 'mass_ratio' in data:
        m2 = m1 * data['mass_ratio']
    else:
        raise KeyError("PlanckWindow_PrimaryMassSecondaryMass_TwoMmin expects 'log_mass_2', 'mass_2', or 'mass_ratio' in data.")

    m1smoothed = m_smoother(m1, mmin_1, delta_m_1)
    m2smoothed = m_smoother(m2, mmin_2, delta_m_2)
    return m1smoothed + m2smoothed

def PlanckWindow_PrimaryMassSecondaryMass(data, mmin, delta_m):
    """
    Planck-taper style window acting on m1 and m2, returned in log-space.
    """
    
    return PlanckWindow_PrimaryMassSecondaryMass_TwoMmin(data, mmin, delta_m, mmin, delta_m)

def BrokenPowerLaw(data, slope_1, slope_2, xmin, xmax, break_fraction):
    """
    Broken power-law distribution with a single spectral break.

    Defines a continuous piecewise power-law across [xmin, xmax] with
    slopes `slope_1` (below the break) and `slope_2` (above the break).
    The break location is determined by `break_fraction` of the interval.

    Parameters
    ----------
    data : jnp.ndarray
        Evaluation points.
    slope_1 : float
        Power-law slope below the break.
    slope_2 : float
        Power-law slope above the break.
    xmin : float
        Lower support bound.
    xmax : float
        Upper support bound.
    break_fraction : float
        Fractional location of the break within [xmin, xmax].

    Returns
    -------
    jnp.ndarray
        Log-probability density of the broken power-law distribution.
    """
    m_break = xmin + break_fraction * (xmax - xmin)
    correction = powerlaw(m_break, slope_2, m_break, xmax) - powerlaw(
        m_break, slope_1, xmin, m_break
    )
    low_part = powerlaw(data, slope_1, xmin, m_break)
    high_part = powerlaw(data, slope_2, m_break, xmax)
    
    # this might be nan gradient?
    prob = jnp.where(data < m_break, low_part + correction, high_part)

    return prob + log_sigmoid(-correction) # - log(1+exp(correction))

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

def trunc_gaussian(data, mean, sig, lower, upper):
    """
    Truncated Gaussian distribution. Numerically stable implementation adapted from
    https://github.com/ColmTalbot/gwpopulation/blob/6e60056be9ae809515eb4576e1ab581c5607a49c/gwpopulation/utils.py#L133-L183

    Parameters
    ----------
    data : jnp.ndarray
        Evaluation points.
    mean : float
        Mean of the Gaussian.
    sig : float
        Standard deviation of the Gaussian.
    lower : float
        Lower truncation bound.
    upper : float
        Upper truncation bound.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the truncated Gaussian,
        with −INF outside [lower, upper].
    """
    
    def logsubexp(log_p, log_q):
        return log_p + jnp.log(1 - jnp.exp(log_q - log_p))

    up = (upper - mean) / sig
    lo = (lower - mean) / sig

    px = -(data - mean)**2 / 2 / sig**2 - np.log(2.0 * np.pi) / 2.0 - jnp.log(sig)

    # cf https://github.com/scipy/scipy/blob/v1.15.1/scipy/stats/_continuous_distns.py#L10189
    log_norm = jnp.select(
        [up <= 0, lo > 0, up > 0],
        [
            logsubexp(scs.log_ndtr(up), scs.log_ndtr(lo)),
            logsubexp(scs.log_ndtr(-lo), scs.log_ndtr(-up)),
            jnp.log1p(-scs.ndtr(lo) - scs.ndtr(-up)),
        ],
        jnp.nan,
    )
    px -= log_norm
    in_support = jnp.logical_and(data < upper, data > lower)
    return jnp.where(in_support, px, -jnp.inf*jnp.ones_like(data))

def chieff_gaussian(data, mean, sig):
    """
    Effective spin distribution: Gaussian in chi_eff.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict containing key 'chi_eff', or direct array of chi_eff values.
    mean : float
        Mean of the Gaussian.
    sig : float
        Standard deviation of the Gaussian.

    Returns
    -------
    jnp.ndarray
        Log-probability density under the Gaussian distribution.
    """
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    return trunc_gaussian(x, mean, sig, -1, 1)


# def chieff_skew_gaussian(data, mu_eff, sig_eff, eps):
    
#     if isinstance(data, dict):
#         x = data['chi_eff']
#     else:
#         x = data

#     tiny = 1e-6
#     eps = jnp.clip(eps, -1.0 + tiny, 1.0 - tiny)
#     sig_eff = jnp.clip(sig_eff, tiny)

#     sig_left = sig_eff * (1.0 + eps)    # x <= 0
#     sig_right = sig_eff * (1.0 - eps)   # x >= 0

#     def log_unnorm(xx):
#         log_left = jnp.log1p(eps) + trunc_gaussian(xx, mu_eff, sig_left, -1.0, 1.0)
#         log_right = jnp.log1p(-eps) + trunc_gaussian(xx, mu_eff, sig_right, -1.0, 1.0)
#         return jnp.where(xx <= 0.0, log_left, log_right)

#     logp = log_unnorm(x)

#     ngrid=2000
#     grid = jnp.linspace(-1.0, 1.0, ngrid)
#     dx = grid[1] - grid[0]

#     logp_grid = log_unnorm(grid)
#     logZ = scs.logsumexp(logp_grid) + jnp.log(dx)

#     return logp - logZ

def chieff_skew_gaussian(data, mu_x, sig_x, eps_x):
    # Version from gwpop code.
    
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data

    tiny = 1e-12
    eps = 1e-6

    sig_x = jnp.clip(sig_x, tiny)
    eps_x = jnp.clip(eps_x, -1.0 + eps, 1.0 - eps)

    low = -1.0
    high = 1.0

    norm = sig_x * jnp.sqrt(jnp.pi / 2.0)
    a = (high - mu_x) / (jnp.sqrt(2.0) * sig_x * (1.0 - eps_x))
    b = (mu_x - low) / (jnp.sqrt(2.0) * sig_x * (1.0 + eps_x))
    norm = norm * ((1.0 - eps_x) * scs.erf(a) + (1.0 + eps_x) * scs.erf(b))

    log_p_left = -((x - mu_x) ** 2) / (2.0 * sig_x**2 * (1.0 + eps_x) ** 2)
    log_p_right = -((x - mu_x) ** 2) / (2.0 * sig_x**2 * (1.0 - eps_x) ** 2)

    log_p = jnp.where(x > mu_x, log_p_right, log_p_left) - jnp.log(norm)

    return jnp.where((x >= low) & (x <= high), log_p, -jnp.inf)

def chieff_two_gaussians(data, mu_x1, sig_x1, mu_x2, sig_x2, lamb_x):
    
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data

    eps = 1e-6
    lamb_x = jnp.clip(lamb_x, eps, 1 - eps)

    # log truncated gaussian components
    log_g1 = trunc_gaussian(x, mu_x1, sig_x1, -1, 1)
    log_g2 = trunc_gaussian(x, mu_x2, sig_x2, -1, 1)

    return jnp.logaddexp(
        jnp.log1p(-lamb_x) + log_g1,
        jnp.log(lamb_x) + log_g2
    )


def gaussian_2d_m1_chieff(data, mu_m1, sig_m1, mu_x, sig_x, rho):
    """
    2D Gaussian in (m1, chi_eff) with correlation.

    The chi_eff marginal is truncated to [-1, 1].  Follows the same
    data / Jacobian convention as BrokenPowerlawPlusTwoPeaks_PrimaryMass:
    accepts 'log_mass_1' or 'mass_1', and adds the log(m1) Jacobian
    when the data is in log space.

    Parameters
    ----------
    data : dict
        Must contain 'chi_eff' and either 'log_mass_1' or 'mass_1'.
    mu_m1 : float
        Mean of m1.
    sig_m1 : float
        Std dev of m1.
    mu_x : float
        Mean of chi_eff.
    sig_x : float
        Std dev of chi_eff.
    rho : float
        Correlation coefficient between m1 and chi_eff, in (-1, 1).

    Returns
    -------
    jnp.ndarray
        Log-probability density evaluated at each event, shape (N,).
    """
    isLogMass = True
    try:
        log_m1 = data['log_mass_1']
        m1 = jnp.exp(log_m1)
    except KeyError:
        isLogMass = False
        m1 = data['mass_1']

    x = data['chi_eff']

    eps = 1e-6
    rho = jnp.clip(rho, -1 + eps, 1 - eps)
    sig_m1 = jnp.clip(sig_m1, eps)
    sig_x = jnp.clip(sig_x, eps)

    z_m = (m1 - mu_m1) / sig_m1
    z_x = (x - mu_x) / sig_x

    # bivariate normal log-pdf in (m1, chi_eff)
    log_p = (
        -0.5 / (1 - rho**2) * (z_m**2 - 2 * rho * z_m * z_x + z_x**2)
        - jnp.log(sig_m1)
        - jnp.log(sig_x)
        - jnp.log(2 * jnp.pi)
        - 0.5 * jnp.log(1 - rho**2)
    )

    # truncation normalisation in chi_eff direction
    # conditional mean & std of chi_eff | m1
    mu_x_cond = mu_x + rho * sig_x / sig_m1 * (m1 - mu_m1)
    sig_x_cond = sig_x * jnp.sqrt(1 - rho**2)

    up = (1.0 - mu_x_cond) / sig_x_cond
    lo = (-1.0 - mu_x_cond) / sig_x_cond

    log_trunc_norm = jnp.log(scs.ndtr(up) - scs.ndtr(lo))

    log_p = log_p - log_trunc_norm

    in_support = (x >= -1.0) & (x <= 1.0)
    log_p = jnp.where(in_support, log_p, -jnp.inf)

    if isLogMass:
        log_p = log_p + log_m1

    return log_p


def chieff_two_gaussians_and_two_masses(data, mu_x1, sig_x1, alpha_1, alpha_2, mmin, break_mass, delta_m_1, lam_fractions, mpp_1, sigpp_1, mpp_2, sigpp_2, mu_m1, sig_m1, mu_x2, sig_x2, rho, lamb_x):

    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data

    eps = 1e-6
    lamb_x = jnp.clip(lamb_x, eps, 1 - eps)

    # component 1: independent mass × chi_eff
    log_g1 = trunc_gaussian(x, mu_x1, sig_x1, -1, 1)
    log_bp2pk_1 = BrokenPowerlawPlusTwoPeaks_PrimaryMass(data, alpha_1, alpha_2, mmin,
                                                         break_mass, delta_m_1, lam_fractions,
                                                         mpp_1, sigpp_1, mpp_2, sigpp_2)

    # component 2: correlated 2D Gaussian in (m1, chi_eff)
    log_gauss2d = gaussian_2d_m1_chieff(data, mu_m1, sig_m1, mu_x2, sig_x2, rho)

    return jnp.logaddexp(
        jnp.log1p(-lamb_x) + log_g1 + log_bp2pk_1,
        jnp.log(lamb_x) + log_gauss2d
    )  


def make_chieff_two_gaussians_m1dep(n_nodes=5, log_m1_min=None, log_m1_max=None,
                                     method='cubic'):
    """Build a chi_eff two-Gaussian model whose branching fraction varies with m1.

    The branching fraction lamb_x(m1) is parameterised at ``n_nodes`` control
    points evenly spaced in log(m1).  Each node value lives in logit space
    and is interpolated with ``interpax.interp1d``, then mapped to (0, 1)
    with a sigmoid.

    Parameters
    ----------
    n_nodes : int
        Number of interpolation control points (default 5).
    log_m1_min, log_m1_max : float or None
        Bounds of the node grid in log(m1).  Defaults: log(2) and log(200).
    method : str
        ``interpax.interp1d`` method (default ``'linear'``).

    Returns
    -------
    chieff_two_gaussians_m1dep : callable
        ``(data, mu_x1, sig_x1, mu_x2, sig_x2,
        logit_lamb_x_0, …, logit_lamb_x_{n-1})  ->  log p(chi_eff|m1)``

    The returned function has attributes:

    * ``hyper_names`` – ordered list of all hyperparameter names, ready for
      ``parameter_to_hyperparameters['chi_eff']``.
    * ``logit_names`` – just the node names (for setting up priors).
    * ``nodes`` – the log(m1) node locations (jnp array).

    Example usage (analogous to two-gaussians-onlypar.py)::

        chieff_m1dep = make_chieff_two_gaussians_m1dep(
            n_nodes=5,
            log_m1_min=jnp.log(3),
            log_m1_max=jnp.log(300),
        )

        mixture_parametric_models = {'chi_eff': chieff_m1dep}

        parameter_to_hyperparameters = {
            ...
            'chi_eff': chieff_m1dep.hyper_names,
        }

        priors = {
            ...
            'mu_x1': ([-1, 0.2], dist.Uniform),
            'sig_x1': ([0.03, 0.6], dist.Uniform),
            'mu_x2': ([0.2, 1.], dist.Uniform),
            'sig_x2': ([0.03, 0.4], dist.Uniform),
            # logit-space node priors (0 ↔ lamb_x=0.5)
            **{k: ([-5., 5.], dist.Uniform) for k in chieff_m1dep.logit_names},
        }
    """
    import interpax as ipx
    from jax.nn import sigmoid

    if log_m1_min is None:
        log_m1_min = np.log(3.)
    if log_m1_max is None:
        log_m1_max = np.log(300.)

    nodes = jnp.linspace(log_m1_min, log_m1_max, n_nodes)
    logit_names = [f'logit_lamb_x_{i}' for i in range(n_nodes)]
    hyper_names = ['mu_x1', 'sig_x1', 'mu_x2', 'sig_x2'] + logit_names

    def chieff_two_gaussians_m1dep(data, mu_x1, sig_x1, mu_x2, sig_x2,
                                   *logit_lamb_nodes):
        if isinstance(data, dict):
            x = data['chi_eff']
            if 'log_mass_1' in data:
                log_m1 = data['log_mass_1']
            else:
                log_m1 = jnp.log(data['mass_1'])
        else:
            raise ValueError(
                "chieff_two_gaussians_m1dep requires a data dict with "
                "'chi_eff' and 'mass_1' or 'log_mass_1'"
            )

        logit_vals = jnp.array(logit_lamb_nodes)

        # Interpolate logit(lamb_x) at each sample's log(m1)
        shape = log_m1.shape
        logit_lamb = ipx.interp1d(
            log_m1.ravel(), nodes, logit_vals,
            method=method, extrap=True,
        )
        logit_lamb = logit_lamb.reshape(shape)

        lamb_x = sigmoid(logit_lamb)
        eps = 1e-6
        lamb_x = jnp.clip(lamb_x, eps, 1 - eps)

        # Truncated Gaussian components
        log_g1 = trunc_gaussian(x, mu_x1, sig_x1, -1, 1)
        log_g2 = trunc_gaussian(x, mu_x2, sig_x2, -1, 1)

        return jnp.logaddexp(
            jnp.log1p(-lamb_x) + log_g1,
            jnp.log(lamb_x) + log_g2
        )

    chieff_two_gaussians_m1dep.__name__ = (
        f'chieff_two_gaussians_m1dep_{n_nodes}nodes'
    )
    chieff_two_gaussians_m1dep.hyper_names = hyper_names
    chieff_two_gaussians_m1dep.logit_names = logit_names
    chieff_two_gaussians_m1dep.nodes = nodes

    return chieff_two_gaussians_m1dep


def chieff_three_gaussians(data, mu_x1, sig_x1, mu_x2, sig_x2, mu_x3, sig_x3, lamb_x1, lamb_x2):
    
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    eps = 1e-6
    lamb_x1 = jnp.clip(lamb_x1, eps, 1 - eps)
    lamb_x2 = jnp.clip(lamb_x2, eps, 1 - eps)
    # Ensure weights sum to 1: w1 = lamb_x1, w2 = (1 - lamb_x1) * lamb_x2, w3 = (1 - lamb_x1) * (1 - lamb_x2)
    log_w1 = jnp.log(lamb_x1)
    log_w2 = jnp.log1p(-lamb_x1) + jnp.log(lamb_x2)
    log_w3 = jnp.log1p(-lamb_x1) + jnp.log1p(-lamb_x2)
    # log truncated gaussian components
    log_g1 = trunc_gaussian(x, mu_x1, sig_x1, -1, 1)
    log_g2 = trunc_gaussian(x, mu_x2, sig_x2, -1, 1)
    log_g3 = trunc_gaussian(x, mu_x3, sig_x3, -1, 1)
    return jnp.logaddexp(
        jnp.logaddexp(log_w1 + log_g1, log_w2 + log_g2),
        log_w3 + log_g3
    )

def chieff_skew_plus_gaussian(data, mu_x1, sig_x1, mu_x2, sig_x2, eps_x, lamb_x):
    
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data

    tiny = 1e-6
    eps_x = jnp.clip(eps_x, -1 + tiny, 1 - tiny)
    lamb_x = jnp.clip(lamb_x, tiny, 1 - tiny)

    # log truncated-gaussian components
    log_g1 = chieff_skew_gaussian(x, mu_x1, sig_x1, eps_x)
    log_g2 = trunc_gaussian(x, mu_x2, sig_x2, -1, 1)

    return jnp.logaddexp(
        jnp.log1p(-lamb_x) + log_g1,
        jnp.log(lamb_x) + log_g2
    )

# BrokenPowerlawPlusTwoPeaks_PrimaryMass(
# data, alpha_1, alpha_2, mmin, break_mass, delta_m_1, 
# lam_fractions, mpp_1, sigpp_1, mpp_2, sigpp_2


def chieff_two_gaussians_and_mass(data, mu_x1, sig_x1, mu_x2, sig_x2, lamb_x, alpha_1_1, alpha_2_1, mmin_1, break_mass_1, delta_m_1_1,lam_fractions_1, mpp_1_1, sigpp_1_1, mpp_2_1, sigpp_2_1, alpha_1_2, alpha_2_2, mmin_2, break_mass_2, delta_m_1_2,lam_fractions_2, mpp_1_2, sigpp_1_2, mpp_2_2, sigpp_2_2):
    
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data

    eps = 1e-6
    lamb_x = jnp.clip(lamb_x, eps, 1 - eps)

    # log truncated gaussian components
    log_g1 = trunc_gaussian(x, mu_x1, sig_x1, -1, 1)
    log_g2 = trunc_gaussian(x, mu_x2, sig_x2, -1, 1)

    #mass parameters
    log_bp2pk_1 = BrokenPowerlawPlusTwoPeaks_PrimaryMass(data, alpha_1_1, alpha_2_1, mmin_1, 
                                                       break_mass_1, delta_m_1_1,lam_fractions_1, 
                                                       mpp_1_1, sigpp_1_1, mpp_2_1, sigpp_2_1)
    log_bp2pk_2 = BrokenPowerlawPlusTwoPeaks_PrimaryMass(data, alpha_1_2, alpha_2_2, mmin_2, 
                                                       break_mass_2, delta_m_1_2,lam_fractions_2, 
                                                       mpp_1_2, sigpp_1_2, mpp_2_2, sigpp_2_2)

    return jnp.logaddexp(
        jnp.log1p(-lamb_x) + log_g1 + log_bp2pk_1,
        jnp.log(lamb_x) + log_g2 + log_bp2pk_2
    )  

# PowerlawPlusPeak_PrimaryMass(data, alpha, minimum, maximum, delta_m, mpp, sigpp, lam)
def chieff_two_gaussians_and_mass_simpler(data, mu_x1, sig_x1, mu_x2, sig_x2, lamb_x, alpha_1_1, alpha_2_1, mmin_1, break_mass_1, delta_m_1_1,lam_fractions_1, mpp_1_1, sigpp_1_1, mpp_2_1, sigpp_2_1, alpha_1_2, mmin_2, mmax_2, delta_m_2, mpp_1_2, sigpp_1_2, lam_2):
        
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data

    eps = 1e-6
    lamb_x = jnp.clip(lamb_x, eps, 1 - eps)

    # log truncated gaussian components
    log_g1 = trunc_gaussian(x, mu_x1, sig_x1, -1, 1)
    log_g2 = trunc_gaussian(x, mu_x2, sig_x2, -1, 1)

    #mass parameters
    log_bp2pk_1 = BrokenPowerlawPlusTwoPeaks_PrimaryMass(data, alpha_1_1, alpha_2_1, mmin_1, 
                                                       break_mass_1, delta_m_1_1,lam_fractions_1, 
                                                       mpp_1_1, sigpp_1_1, mpp_2_1, sigpp_2_1)
    log_plpk = PowerlawPlusPeak_PrimaryMass(data, alpha_1_2, mmin_2, mmax_2, delta_m_2, mpp_1_2,
                                            sigpp_1_2, lam_2)

    return jnp.logaddexp(
        jnp.log1p(-lamb_x) + log_g1 + log_bp2pk_1,
        jnp.log(lamb_x) + log_g2 + log_plpk
    )  

    

def nothing(data):
    return jnp.log(1)


def chip_gaussian(data, mean, sig):
    """
    Effective precessing spin distribution: Gaussian in chi_p.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict containing key 'chi_p', or direct array of chi_p values.
    mean : float
        Mean of the Gaussian.
    sig : float
        Standard deviation of the Gaussian.

    Returns
    -------
    jnp.ndarray
        Log-probability density under the Gaussian distribution.
    """
    if isinstance(data, dict):
        x = data['chi_p']
    else:
        x = data
    return trunc_gaussian(x, mean, sig, 0, 1)

def lognormal(data, mean, sig):
    """
    Log-normal distribution.

    Parameters
    ----------
    data : jnp.ndarray
        Evaluation points (must be > 0).
    mean : float
        location parameter of lognormal (mean  of ln(X) if X~LogNormal)
    sig : float
        width parameter of lognormal (standard deviation of ln(X) if X~LogNormal)

    Returns
    -------
    jnp.ndarray
        Log-probability density of the log-normal distribution.
    """
    px = -(jnp.log(data) - mean)**2 / 2 / sig**2
    denom = jnp.log(data*sig*jnp.sqrt(2*jnp.pi))
    return px - denom

# TODO: base Redshift function that takes in Psi evolution and returns log density
def PowerlawRedshift(data, lamb, max_z=1.9, normalize=True, return_normalization=False):
    """
    Redshift distribution model: power law in (1+z) weighted by comoving volume.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict containing key 'redshift', or direct array of redshifts.
    lamb : float
        Power-law index on (1+z).
    max_z : float, optional
        Maximum redshift cutoff (default 1.9).
    normalize : bool, optional
        If True, normalize the distribution (default True).
    return_normalization : bool, optional
        If True, return only the log-normalization constant.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the redshift distribution.
        If `return_normalization=True`, returns the log-normalization only.
    """
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    zs_fixed = jnp.linspace(1e-5, max_z, 1000)
    dvs = COSMO.differential_comoving_volume(zs_fixed)
    if isinstance(dvs, unxt.quantity.Quantity):
        # TODO: preferably would use unxt.unit values...
        dvs = 4*jnp.pi * 1e-9 * dvs.value
    else:
        dvs = 4*jnp.pi * 1e-9 * dvs
    fixed_ln_dvc_dz = jnp.log(dvs)

    if normalize:
        dz = zs_fixed[1] - zs_fixed[0]
        test_ln_p = fixed_ln_dvc_dz + (lamb - 1) * jnp.log(1. + zs_fixed)
        ln_norm = scs.logsumexp(test_ln_p) + jnp.log(dz)
        if return_normalization:
            return ln_norm
    else:
        ln_norm = 0.
    ln_dvc_dz = jnp.interp(z, zs_fixed, fixed_ln_dvc_dz)
    ln_p = ln_dvc_dz + (lamb - 1) * jnp.log(1. + z)
    ln_p -= ln_norm
        
    window = jnp.logical_and(z >= 0., z <= max_z)
    p = jnp.where(window, ln_p, -INF*jnp.ones_like(z))
    return p

def PowerlawRedshiftPsi(data, lamb, max_z=1.9):
    """
    Power-law redshift distribution: proportional to (1+z)^lamb.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict containing key 'redshift', or direct array of redshifts.
    lamb : float
        Power-law index on (1+z).
    max_z : float, optional
        Maximum redshift cutoff (default 1.9).

    Returns
    -------
    jnp.ndarray
        Log-probability density, with −INF outside [0, max_z].
    """
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    ln_p = lamb * jnp.log(1. + z)

    window = jnp.logical_and(z >= 0., z <= max_z)
    p = jnp.where(window, ln_p, -INF*jnp.ones_like(z))
    return p

def MadauDickinsonRedshift(data, gamma, kappa, z_peak, z_max=1.9, normalize=True, return_normalization=False):
    """
    Madau–Dickinson star-formation rate redshift distribution.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict containing key 'redshift', or direct array of redshifts.
    gamma : float
        Low-redshift power-law index.
    kappa : float
        High-redshift suppression exponent.
    z_peak : float
        Characteristic peak redshift.
    z_max : float, optional
        Maximum redshift cutoff (default 1.9).
    normalize : bool, optional
        If True, normalize the distribution (default True).
    return_normalization : bool, optional
        If True, return only the log-normalization constant.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the redshift distribution.
        If `return_normalization=True`, returns the log-normalization only.
    """
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    zs_fixed = np.linspace(1e-5, z_max, 1000)
    fixed_ln_dvc_dz = jnp.log(
        4*jnp.pi*COSMO.differential_comoving_volume(zs_fixed).to(unxt.Gpc**3 / unxt.sr).value
        )
    if normalize:
        dz = zs_fixed[1] - zs_fixed[0]
        test_ln_p = fixed_ln_dvc_dz + (gamma - 1)* jnp.log(1. + zs_fixed) - jnp.log(1 + ((1 + zs_fixed)/(1 + z_peak))**kappa)
        ln_norm = scs.logsumexp(test_ln_p) + jnp.log(dz)
        if return_normalization:
            return ln_norm
    else:
        ln_norm = 0.
    ln_dvc_dz = jnp.interp(z, zs_fixed, fixed_ln_dvc_dz)
    ln_p = ln_dvc_dz + (gamma - 1)* jnp.log(1. + z) - jnp.log(1 + ((1 + z)/(1 + z_peak))**kappa)
    ln_p -= ln_norm

    window = jnp.logical_and(z >= 0., z <= z_max)
    p = jnp.where(window, ln_p, -INF*jnp.ones_like(z))
    return p

def PowerlawPlusPeak_MassRatio(data, slope, minimum, delta_m):
    """
    Mass-ratio distribution: smoothed power law with minimum mass cut.

    Parameters
    ----------
    data : dict
        Must contain 'mass_ratio' and either 'mass_1' or 'log_mass_1'.
    slope : float
        Power-law slope on the mass ratio q.
    minimum : float
        Global minimum BH mass.
    delta_m : float
        Mass smoothing scale at the minimum cutoff.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the smoothed mass-ratio distribution.
    """

    try:
        m1 = jnp.exp(data['log_mass_1'])
    except KeyError:
        m1 = data['mass_1']
    q = data['mass_ratio']

    power_law = powerlaw(q, slope, minimum/m1, jnp.ones_like(m1))
    smoothed_pl = power_law + m_smoother(q*m1, minimum, delta_m)

    m1s_test = jnp.exp(jnp.linspace(jnp.log(2.), jnp.log(100.), 500))
    m2s_test = jnp.linspace(1.99*jnp.ones_like(m1s_test), m1s_test, 10000)
    qs_test = m2s_test / jnp.expand_dims(m1s_test, axis=0)
    dq = qs_test[1] - qs_test[0]
    power_law_test = powerlaw(qs_test, slope, 0.02, 1.) # fiducial lower bound of 0.02 
    smoothed_pl_test = power_law_test + m_smoother(m2s_test, minimum, delta_m)
    
    norm = scs.logsumexp(smoothed_pl_test, axis=0) + jnp.log(dq) # simple Riemann rule
    # norms = jnp.interp(m1, m1s_test, norm)
    norms = norm[jnp.digitize(m1, m1s_test)] # take the point to the right of each m1, so
    # that the normalization is always SMALLER than the true value, so that 
    # correct normalization from fiducial lower bound
    norms += jnp.log(jnp.abs(1 - 0.02**(slope+1))) - jnp.log(jnp.abs(1 - (minimum/m1)**(slope+1)))
    return smoothed_pl - norms

def Powerlaw_MassRatio(data, slope, minimum):
    """
    Simple power-law mass-ratio distribution with a global minimum mass.

    Parameters
    ----------
    data : dict
        Must contain 'mass_ratio' and either 'mass_1' or 'log_mass_1'.
    slope : float
        Power-law slope on q.
    minimum : float
        Global minimum BH mass.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the mass-ratio distribution.
    """
    try:
        m1 = jnp.exp(data['log_mass_1'])
    except KeyError:
        m1 = data['mass_1']
    q = data['mass_ratio']

    power_law = powerlaw(q, slope, minimum/m1, jnp.ones_like(m1))
    return power_law

def SimplePowerlaw_MassRatio(data, slope, qmin):
    """
    Simple power-law mass-ratio distribution without a global minimum mass.

    Parameters
    ----------
    data : dict
        Must contain 'mass_ratio'.
    slope : float
        Power-law slope on q.
    qmin : float
        Minimum mass ratio allowed.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the mass-ratio distribution.
    """
    q = data['mass_ratio']

    power_law = powerlaw(q, slope, qmin, 1.)
    return power_law


def PowerlawPlusPeak(data, alpha, beta, mmin, mmax, delta_m, mpp, sigpp, lam):
    """
    Joint primary-mass and mass-ratio distribution: power law plus Gaussian peak.

    Parameters
    ----------
    data : dict
        Must contain 'mass_1' or 'log_mass_1', and 'mass_ratio'.
    alpha : float
        Power-law slope for primary mass.
    beta : float
        Power-law slope for mass ratio.
    mmin, mmax : float
        Minimum and maximum primary masses.
    delta_m : float
        Smoothing scale at lower mass cutoff.
    mpp : float
        Peak mass location.
    sigpp : float
        Standard deviation of the peak Gaussian.
    lam : float
        Fraction in the Gaussian peak.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the joint distribution.
    """

    pm1 = PowerlawPlusPeak_PrimaryMass(data, alpha, mmin, mmax, delta_m, mpp, sigpp, lam)
    pq = PowerlawPlusPeak_MassRatio(data, beta, mmin, delta_m)

    return pm1 + pq

def smooth(x, cutoff, width):
    """
    Smooth cutoff function with continuous derivative.

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points.
    cutoff : float
        Cutoff location.
    width : float
        Width of smoothing region.

    Returns
    -------
    jnp.ndarray
        Smooth step function, transitioning quadratically at cutoff.
    """
    return jnp.where(x<cutoff, 0., -((x-cutoff)/width)**2)

def mu_var_to_alpha_beta(mu, var):
    """
    Convert mean and variance to Beta distribution parameters.

    Parameters
    ----------
    mu : float
        Mean of the Beta distribution.
    var : float
        Variance of the Beta distribution.

    Returns
    -------
    alpha : float
        Beta shape parameter alpha.
    beta : float
        Beta shape parameter beta.
    """
    nu = (mu*(1-mu)/var) - 1
    alpha = mu * nu
    beta = (1-mu) * nu
    return alpha, beta

def beta_spin(spin_mag, alpha, beta):
    """
    Beta distribution for spin magnitudes.

    Parameters
    ----------
    spin_mag : jnp.ndarray
        Spin magnitudes in [0, 1].
    alpha : float
        Beta distribution parameter alpha.
    beta : float
        Beta distribution parameter beta.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the Beta distribution.
    """
    ln_a = jnp.log(spin_mag)
    ln_1ma = jnp.log(1. - spin_mag)
    ln_p = (alpha - 1) * ln_a + (beta - 1) * ln_1ma

    norm = scs.gammaln(alpha) + scs.gammaln(beta) - scs.gammaln(alpha + beta)
    return ln_p - norm

def beta_spin_mv(spin_mag, mu, var):
    """
    Beta distribution for spin magnitudes.

    Parameters
    ----------
    spin_mag : jnp.ndarray
        Spin magnitudes in [0, 1].
    mu : float
        Beta distribution mean.
    var : float
        Beta distribution variance.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the Beta distribution.
    """
    alpha, beta = mu_var_to_alpha_beta(mu, var)
    return beta_spin(spin_mag, alpha, beta)

def iid_beta_spin(data, mu, var):
    """
    Beta distribution for spin magnitudes.

    Parameters
    ----------
    data : dict
        Must contain 'a_1' and 'a_2'
    mu : float
        Beta distribution mean.
    var : float
        Beta distribution variance.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the Beta distribution.
    """
    alpha, beta = mu_var_to_alpha_beta(mu, var)
    return beta_spin(data['a_1'], alpha, beta) + beta_spin(data['a_2'], alpha, beta)

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
    return trunc_gaussian(data['a_1'], mu, sig, 0, 1) + trunc_gaussian(data['a_2'], mu, sig, 0, 1)

def iid_normal_spin_fms(data, mu, var, NS_amax=0.4, NS_mmax=2.5):
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
    keys = data.keys()
    if 'mass_1' in keys:
        m1 = data['mass_1']
    elif 'log_mass_1' in keys:
        m1 = jnp.exp(data['log_mass_1'])
    if 'mass_ratio' in keys:
        m2 = m1 * data['mass_ratio']
    elif 'log_mass_2' in keys:
        m2 = jnp.exp(data['log_mass_2'])
    regions = {'mass_1': m1, 'mass_2': m2}
    for ii in [1,2]:
        probs = jnp.where(
            regions[f'mass_{ii}'] < NS_mmax, 
            trunc_gaussian(data[f'a_{ii}'], mu, sig, 0, NS_amax), 
            trunc_gaussian(data[f'a_{ii}'], mu, sig, 0, 1)
            )
        total_prob += probs
    
    return total_prob

def tilt_model(data, mu, sig, zeta):
    """
    Tilt distribution model allowing a free mean tilt parameter.

    Models the joint distribution of the cosine tilts of both black holes
    (`cos_tilt_1`, `cos_tilt_2`) as either:
      - a truncated Gaussian centered at `mu` with width `sig`, with probability `zeta`, or
      - an isotropic distribution, with probability `1 - zeta`.

    Parameters
    ----------
    data : dict
        Must contain 'cos_tilt_1' and 'cos_tilt_2'.
    mu : float
        Mean of the truncated Gaussian.
    sig : float
        Standard deviation of the truncated Gaussian.
    zeta : float
        Mixture fraction for the field (truncated Gaussian) component.

    Returns
    -------
    jnp.ndarray
        Log-probabilities of the tilt distribution.
    """
    pfield1 = trunc_gaussian(data['cos_tilt_1'], mu, sig, -1, 1)
    pfield2 = trunc_gaussian(data['cos_tilt_2'], mu, sig, -1, 1)

    pisotropic = jnp.log(jnp.ones_like(data['cos_tilt_1']) / 4)
    pfield = pfield1 + pfield2

    ln_zeta = jnp.log(zeta)
    ln_1mzeta = jnp.log(1 - zeta)

    return jnp.logaddexp(ln_zeta + pfield, ln_1mzeta + pisotropic)


def tilt_default(data, sig, zeta):
    """
    Default tilt distribution model.

    Assumes the tilt distribution is not independent across components:
    either both tilts are isotropic or both follow a truncated Gaussian
    centered at `mu=1`.

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
    return tilt_model(data, 1., sig, zeta)


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
    pfield1 = trunc_gaussian(data['cos_tilt_1'], mu, sig, -1, 1)
    pfield2 = trunc_gaussian(data['cos_tilt_2'], mu, sig, -1, 1)

    pisotropic = jnp.log(jnp.ones_like(data['cos_tilt_1']) / 2)
    
    ln_zeta = jnp.log(zeta)
    ln_1mzeta = jnp.log(1 - zeta)

    p1 = jnp.logaddexp(ln_zeta + pfield1, ln_1mzeta + pisotropic)
    p2 = jnp.logaddexp(ln_zeta + pfield2, ln_1mzeta + pisotropic)
    return p1 + p2
    

def spin_iid(data, mu, var, mu_tilt, sig_tilt, zeta):
    return iid_normal_spin(data, mu, var) + tilt_iid(data, mu_tilt, sig_tilt, zeta)

def gwtc3_spin_default(data, mu, var, sig_tilt, zeta):
    return iid_beta_spin(data, mu, var) + tilt_default(data, sig_tilt, zeta)

def spin_default(data, mu, var, sig_tilt, zeta):
    return iid_beta_spin(data, mu, var) + tilt_default(data, sig_tilt, zeta)

@partial(jit, static_argnames=['rate_likelihood','return_likelihood_info'])
def hierarchical_likelihood(event_weights, denominator_weights, total_injections, live_time=1, rate_likelihood=False, return_likelihood_info=True):
    """
    Hierarchical Bayesian likelihood for population inference.

    Parameters
    ----------
    event_weights : jnp.ndarray
        Array (n_events, n_samples) of log[p(θ|pop)/pi(θ|PE)] for each event posterior sample.
    denominator_weights : jnp.ndarray
        Array of log[p(θ|pop)/pi(θ|draw)] for injections.
    total_injections : int
        Number of injection samples.
    live_time : float, optional
        Observing time in years (default 1).
    rate_likelihood : bool, optional
        If True, include rate likelihood (default False).
    return_likelihood_info : bool, optional
        If True, return decomposition of likelihood and variances.

    Returns
    -------
    tuple
        If `return_likelihood_info=True`:
            (lnL, var, [pe_lnL, vt_lnL], [pe_var, vt_var])
        else:
            (lnL, var)
    """
    n_events, minimum_length = event_weights.shape
    numerators = scs.logsumexp(event_weights, axis=1) - jnp.log(minimum_length) # means
    denominator = scs.logsumexp(denominator_weights) - jnp.log(total_injections)

    pe_ln_likelihood = jnp.sum(numerators)
    if rate_likelihood:
        vt_ln_likelihood = n_events*jnp.log(live_time) - live_time*jnp.exp(denominator)
    else:
        vt_ln_likelihood = -n_events*denominator

    ln_likelihood = pe_ln_likelihood + vt_ln_likelihood
    
    square_sums = scs.logsumexp(2*event_weights, axis=1) - 2*jnp.log(minimum_length) # square_sums
    square_sum = scs.logsumexp(2*denominator_weights) - 2*jnp.log(total_injections)
    
    pe_ln_likelihood_variance = jnp.sum(jnp.exp(square_sums - 2*numerators) - 1/minimum_length)
    if rate_likelihood:
        vt_ln_likelihood_variance = live_time**2 * (jnp.exp(square_sum) - jnp.exp(2*denominator)/total_injections)
    else:
        vt_ln_likelihood_variance = n_events**2 * (jnp.exp(square_sum - 2*denominator) - 1/total_injections)
    
    ln_likelihood_variance = pe_ln_likelihood_variance + vt_ln_likelihood_variance
    
    if return_likelihood_info:
        ln_likelihoods = [pe_ln_likelihood, vt_ln_likelihood]
        ln_likelihood_variances = [pe_ln_likelihood_variance, vt_ln_likelihood_variance]
        return ln_likelihood, ln_likelihood_variance, ln_likelihoods, ln_likelihood_variances
    else:
        return ln_likelihood, ln_likelihood_variance

def rate_likelihood(event_weights, denominator_weights, total_injections, live_time=1):
    """
    Poisson rate likelihood for hierarchical inference.

    Parameters
    ----------
    event_weights : jnp.ndarray
        Array (n_events, n_samples) of log[p(θ|pop)/pi(θ|PE)] for each event posterior sample.
    denominator_weights : jnp.ndarray
        Array of log[p(θ|pop)/pi(θ|draw)] for injections.
    total_injections : int
        Number of injection samples.
    live_time : float, optional
        Observing time in years (default 1).

    Returns
    -------
    tuple
        (lnL, expected_events, pe_var, vt_var, total_var)
    """
    n_events, minimum_length = event_weights.shape
    numerators = scs.logsumexp(event_weights, axis=1) - jnp.log(minimum_length) # means
    denominator = scs.logsumexp(denominator_weights) - jnp.log(total_injections)

    pe_ln_likelihood = jnp.sum(numerators)

    nexp = live_time*jnp.exp(denominator)
    vt_ln_likelihood = n_events*jnp.log(live_time) - nexp
    ln_likelihood = pe_ln_likelihood + vt_ln_likelihood
    
    square_sums = scs.logsumexp(2*event_weights, axis=1) - 2*jnp.log(minimum_length) # square_sums
    square_sum = scs.logsumexp(2*denominator_weights) - 2*jnp.log(total_injections)
    
    pe_ln_likelihood_variance = jnp.sum(jnp.exp(square_sums - 2*numerators) - 1/minimum_length)
    vt_ln_likelihood_variance = live_time**2 * (jnp.exp(square_sum) - jnp.exp(2*denominator)/total_injections)
    
    ln_likelihood_variance = pe_ln_likelihood_variance + vt_ln_likelihood_variance
    return ln_likelihood, nexp, pe_ln_likelihood_variance, vt_ln_likelihood_variance, ln_likelihood_variance


bbh_minima = {
    'log_mass_1': jnp.log(3),
    'mass_1': 3.,
    'mass_2': 3.,
    'mass_ratio': 0.,
    'log_mass_2': jnp.log(3),
    'chi_eff': -1.,
    'chi_p': 0.,
    'redshift': 0.,
    # Use the same support for the auxiliary window parameter as for log_mass_1.
    'log_mass_1_window': jnp.log(3),
}
bbh_maxima = {
    'log_mass_1': jnp.log(200),
    'mass_1': 200.,
    'mass_2': 200.,
    'mass_ratio': 1.,
    'log_mass_2': jnp.log(200),
    'chi_eff': 1.,
    'chi_p': 1.,
    'redshift': 1.9,
    'log_mass_1_window': jnp.log(200),
}


gwparameter_to_model = {
    'mass_1': PowerlawPlusPeak_PrimaryMass, #(data, slope, minimum, maximum, delta_m, mpp, sigpp, lam)
    'log_mass_1': PowerlawPlusPeak_PrimaryMass, #(data, slope, minimum, maximum, delta_m, mpp, sigpp, lam)
    'mass_ratio': SimplePowerlaw_MassRatio, #(data, slope)
    'redshift': PowerlawRedshiftPsi, #(data, lamb, maximum):
    'chi_eff': chieff_gaussian, #(data, mean, sig)
    'chi_p': chip_gaussian,  #(data, mean, sig)
    'spin': spin_default, #(data, mu, var, sig, zeta)
    'a': iid_normal_spin, #(data, mu, var)
    't': tilt_iid, #(data, mu, sig, zeta)
}

typical_hyperparameters = {
    'alpha':3, 'beta':2, 'mmin':2, 'mmax':199, 'delta_m':5, 'mpp':35, 'sigpp':5, 
    'lam':0.005, 'lamb':2, 'mu_x':0.06, 'sig_x':0.1, 'mu_xp':0.3, 'sig_xp':0.2, 'mu_spin':0.2, 'var_spin':0.1, 
    'mu_tilt':0.6, 'sig_tilt':0.6, 'zeta_tilt':0.5, 'lnsigma':-1, 'lncor': -5, 
    'mean': 0, 'qmin': 0.02, 'max_z': 1.9,
}

parameter_values = {
    'mass_1': 40., 'log_mass_1': np.log(40.), 'mass_ratio': 0.9, 'chi_eff': 0., 'chi_p': 0., 'redshift': 0.2, 
    'a_1': 0.2, 'a_2': 0.2, 'cos_tilt_1': 0., 'cos_tilt_2': 0.
    }

gwparameter_to_hyperparameters = {
    'mass_1': ['alpha', 'mmin', 'mmax', 'delta_m', 'mpp', 'sigpp', 'lam'], 
    'log_mass_1': ['alpha', 'mmin', 'mmax', 'delta_m', 'mpp', 'sigpp', 'lam'], 
    'mass_ratio': ['beta', 'qmin'], 
    'redshift': ['lamb', 'max_z'],
    'redshift_psi': ['lamb', 'max_z'],
    'chi_eff': ['mu_x', 'sig_x'], 
    'chi_p': ['mu_xp', 'sig_xp'], 
    'spin': ['mu_spin', 'var_spin', 'sig_tilt', 'zeta_tilt'], 
    'a': ['mu_spin', 'var_spin'],
    't': ['mu_tilt', 'sig_tilt', 'zeta_tilt'],
}

default_priors = {
    'alpha': ([-4, 12], dist.Uniform), 
    'beta': ([-2, 7], dist.Uniform), 
    'qmin': ([0, 1], dist.Uniform), 
    'mmin': ([2, 10], dist.Uniform), 
    'mmax': ([60, 200], dist.Uniform), 
    'delta_m': ([0, 10], dist.Uniform), 
    'mpp': ([20, 50], dist.Uniform), 
    'sigpp': ([1, 10], dist.Uniform), 
    'lam': ([0, 1], dist.Uniform), 
    'lamb': ([-2, 10], dist.Uniform), 
    'mu_x': ([-1, 1], dist.Uniform), 
    'sig_x': ([0.005, 1.], dist.Uniform), 
    'mu_xp': ([0, 1], dist.Uniform), 
    'sig_xp': ([0.005, 1.], dist.Uniform), 
    'mu_spin': ([0, 1], dist.Uniform),
    'var_spin': ([0.005, 0.25], dist.Uniform), 
    'mu_tilt': ([-1, 1], dist.Uniform), 
    'sig_tilt': ([0.1, 4], dist.Uniform), 
    'zeta_tilt': ([0, 1], dist.Uniform), 
    'z_minimum': ([0.], dist.Delta), 
    'max_z': ([1.9], dist.Delta),
}

map_to_gwpop_parameters = {
    'mass_1': ['mass_1'], 'log_mass_1': ['log_mass_1'], 'mass_2': ['mass_2'], 'log_mass_2': ['log_mass_2'], 
    'mass_ratio': ['mass_ratio'], 'redshift': ['redshift'], 'redshift_psi': ['redshift_psi'], 'chi_eff': ['chi_eff'], 
    'a_1': ['a_1'], 'a_2': ['a_2'], 'cos_tilt_1': ['cos_tilt_1'], 'cos_tilt_2': ['cos_tilt_2'], 
    'spin': ['a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2'], 'a': ['a_1', 'a_2'], 't': ['cos_tilt_1', 'cos_tilt_2'],
}