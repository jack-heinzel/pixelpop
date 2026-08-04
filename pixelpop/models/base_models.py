"""
Generic, catalog-agnostic building blocks shared by every population model.

These live in their own module so that the catalog-specific model files
(:mod:`~pixelpop.models.O4_models`, :mod:`~pixelpop.models.gwpop_models`) can
both import them without a circular import. Everything here is re-exported by
``gwpop_models``, so ``from pixelpop.models import powerlaw`` and
``from pixelpop.models.gwpop_models import powerlaw`` both keep working.

Every distribution in pixelpop returns a **log** probability density. Products
of densities are sums, and mixtures are ``logaddexp``/``logsumexp``. This keeps
the dynamic range manageable and, unlike working in linear probability, does not
need float64 to resolve the tails.
"""
import jax.numpy as jnp
import jax.scipy.special as scs
import numpy as np
from jax.nn import log_sigmoid

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

    return jnp.where(jnp.isclose(delta, 0),
        jnp.where(m1s >= minimum, 0.0, -INF),
        log_expit(-delta/m_prime - delta/(m_prime - delta))
    )

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

def trunc_gaussian(data, mean, sig, lower, upper):
    """
    Truncated Gaussian distribution. Numerically stable implementation adapted from
    https://github.com/ColmTalbot/gwpopulation/blob/6e60056be9ae809515eb4576e1ab581c5607a49c/gwpopulation/utils.py#L133-L183

    Keep ``lower``/``upper`` scalar: ``jnp.select`` evaluates every branch under
    XLA, so a data-shaped bound costs ~28x a compile-time constant one.

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

def TripleBrokenPowerLaw(data, slope_1, slope_2, slope_3, xmin, xmax,
                         break_1, break_2):
    r"""
    Power-law distribution with two spectral breaks.

    The two-break generalisation of :func:`BrokenPowerLaw`, matching
    ``gwtc6_population_models.mass.triple_power_law_primary_mass``:

    .. math::
        p(x) \propto
        \begin{cases}
            x^{s_1} & x_{\min} \leq x < x_{\rm{b,1}}\\
            x^{s_2} & x_{\rm{b,1}} \leq x < x_{\rm{b,2}}\\
            x^{s_3} & x_{\rm{b,2}} \leq x < x_{\max}
        \end{cases}

    Breaks are absolute positions, not fractions of ``[xmin, xmax]``. Requires
    ``xmin < break_1 < break_2 < xmax``.

    Parameters
    ----------
    data : jnp.ndarray
        Evaluation points.
    slope_1, slope_2, slope_3 : float
        Power-law slopes in the low, middle and high regions.
    xmin, xmax : float
        Support bounds.
    break_1, break_2 : float
        Positions of the first and second breaks.

    Returns
    -------
    jnp.ndarray
        Log-probability density of the twice-broken power-law.
    """
    # Continuity corrections, in log space: correction_1 rescales the low region
    # to meet the middle one at break_1, correction_2 the middle to meet the high
    # one at break_2. Each region is individually normalized by `powerlaw`, so the
    # corrections are exactly the log-ratios at the break points.
    correction_1 = (powerlaw(break_1, slope_2, break_1, break_2)
                    - powerlaw(break_1, slope_1, xmin, break_1))
    correction_2 = (powerlaw(break_2, slope_3, break_2, xmax)
                    - powerlaw(break_2, slope_2, break_1, break_2))

    low_part = powerlaw(data, slope_1, xmin, break_1)
    mid_part = powerlaw(data, slope_2, break_1, break_2)
    high_part = powerlaw(data, slope_3, break_2, xmax)

    prob = jnp.where(
        data < break_1,
        low_part + correction_1 + correction_2,
        jnp.where(data < break_2, mid_part + correction_2, high_part),
    )

    # The three regions carry weight (c1*c2, c2, 1) relative to the high region.
    norm = scs.logsumexp(jnp.array([correction_1 + correction_2, correction_2, 0.0]))

    return prob - norm

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
    return jnp.where(x<cutoff, jnp.zeros_like(x), -((x-cutoff)/width)**2)
