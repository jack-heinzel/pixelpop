"""
Population models, the hierarchical likelihood, and the global default registries.

The models are split across three modules, all re-exported from here so that
``from pixelpop.models.gwpop_models import <anything>`` keeps working:

``base_models``
    Catalog-agnostic primitives -- ``powerlaw``, ``gaussian``, ``trunc_gaussian``,
    ``m_smoother``, ``BrokenPowerLaw``, ``TripleBrokenPowerLaw``.
``O4_models``
    O4-era (GWTC-4 onwards) named population models.
this module
    O3/GWTC-3-era models, redshift models, mass-ratio models, window functions,
    the hierarchical likelihood, and the ``gwparameter_to_model`` /
    ``gwparameter_to_hyperparameters`` / ``default_priors`` registries that
    ``PixelPopData`` falls back on.

The registries below are the **GWTC-4** default set. Per-catalog default sets
(including these) live in :mod:`~pixelpop.models.gwtc_defaults`.
"""
import wcosmo
import unxt
from jax import jit#, lax
from numpyro import distributions as dist
import jax.numpy as jnp
import jax.scipy.special as scs
import numpy as np
from functools import partial

from .base_models import *
from .base_models import INF
from .O4_models import *

Planck15_LAL = wcosmo.FlatLambdaCDM(H0=67.90, Om0=0.3065, name="Planck15_LAL")
COSMO = Planck15_LAL

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

    def _nonorm_plp(m):
        power_law = powerlaw(m, slope, minimum, maximum)
        peak = gaussian(m, mpp, sigpp)
        p = jnp.logaddexp(power_law + jnp.log(1-lam), peak + jnp.log(lam)) + m_smoother(m, minimum, delta_m)
        return p

    m1s_test = jnp.linspace(2.0, 200., 2000)
    dm1 = m1s_test[1] - m1s_test[0]

    pm1 = _nonorm_plp(m1)    
    plp_test = _nonorm_plp(m1s_test)
    
    pm1 -= scs.logsumexp(plp_test) + jnp.log(dm1) # simple Riemann rule

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

def PlanckWindow_SecondaryMass(data, mmin, delta_m):
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
        containing at least 'log_mass_2' or 'mass_2' or 'mass_ratio' and 
        'mass_1' or 'log_mass_1'. 
        When called from ``save_popsummary`` it may instead be a dict with key
        'log_mass_2_window' and a 1D grid.
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
        if 'log_mass_2' in data:
            x = jnp.exp(data['log_mass_2'])
        elif 'mass_2' in data:
            x = data['mass_2']
        elif 'mass_ratio' in data:
            q = data['mass_ratio']
            if 'mass_1' in data:
                m = data['mass_1']
            elif 'log_mass_1' in data:
                m = jnp.exp(data['log_mass_1'])
            else:
                raise KeyError("PlanckWindow_SecondaryMass has 'mass_ratio', and expects 'log_mass_1', or 'mass_1' in data.")
            x = m * q
        else:
            raise KeyError("PlanckWindow_SecondaryMass expects 'log_mass_2', 'mass_2', or 'mass_ratio' in data.")
    else:
        # Assume data is already mass_1.
        x = data

    return m_smoother(x, mmin, delta_m)

def PlanckWindow_MassRatio(data, qmin, delta_q):
    if isinstance(data, dict):
        if 'mass_ratio' in data:
            x = data['mass_ratio']
        else:
            raise KeyError("PlanckWindow_MassRatio expects 'mass_ratio' in data.")
    else:
        # Assume data is already mass_ratio.
        x = data

    # m_smoother clips at a fraction of delta_q, so it already scales to a variable
    # running over [0, 1] and needs no buffer override here.
    return m_smoother(x, qmin, delta_q)

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
    r"""
    Mass-ratio distribution: smoothed power law with minimum mass cut.

    The normalization is performed in two steps to maintain computational efficiency:

    1. Numerical Integration: Computed on a static fiducial grid with $q_{\min} = 0.02$.
    2. Rescaling: Since the power law in the data uses $q_{\min} = m_{\min} / m_1$, we 
    rescale the normalization from the fiducial grid to the physical value.

    We define the target PDF as:
    $$p(q) = \frac{q^{\beta} S(m_1 q \mid m_{\min}, \delta_m)}{\mathcal{I}}$$

    Where the unnormalized density in the code is:
    $$p_{\text{unnorm}} = \text{PL}(q \mid \beta, q_{\min} = \frac{m_{\min}}{m_1}) \times S(m_1 q \mid m_{\min}, \delta_m)$$
    $$p_{\text{unnorm}} = \frac{q^{\beta} S(m_1 q \mid m_{\min}, \delta_m)}{Z(m_{\min}/m_1)}$$

    The true normalization $\mathcal{I}$ is related to the numerical integral over the 
    fiducial grid ($\mathcal{I}_{\text{num}}$) by:
    $$\mathcal{I} = Z(0.02) \times \int_{0.02}^{1} \text{PL}(q \mid \beta, 0.02) S(m_1 q \mid m_{\min}, \delta_m) dq$$
    $$\mathcal{I} = Z(0.02) \times \mathcal{I}_{\text{num}}$$

    Therefore, the final normalized probability is:
    $$p(q) = \frac{p_{\text{unnorm}} \times Z(m_{\min}/m_1)}{Z(0.02) \times \mathcal{I}_{\text{num}}}$$

    where $Z(x) = \frac{1 - x^{\beta+1}}{\beta+1}$.
    
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
        
    else:
        pfield1 = trunc_gaussian(data['cos_tilt_1'], mu, sig, -1, 1)
        pfield2 = trunc_gaussian(data['cos_tilt_2'], mu, sig, -1, 1)

        pisotropic = jnp.log(jnp.ones_like(data['cos_tilt_1']) / 4)
        pfield = pfield1 + pfield2

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


def spin_iid(data, mu, var, mu_tilt, sig_tilt, zeta):
    return iid_normal_spin(data, mu, var) + tilt_iid(data, mu_tilt, sig_tilt, zeta)

def gwtc3_spin_default(data, mu, var, sig_tilt, zeta):
    return iid_beta_spin(data, mu, var) + tilt_default(data, sig_tilt, zeta)

def spin_default(data, mu, var, sig_tilt, zeta):
    return iid_beta_spin(data, mu, var) + tilt_default(data, sig_tilt, zeta)

def _per_event_moments(event_weights, event_counts=None):
    """
    Per-event log-mean and log-mean-square of ``exp(weights)``.

    Accepts a rectangular ``(n_events, n_samples)`` array (the usual case) or, as a
    harmless fallback, a ragged ``list``/``tuple`` of 1-D per-event weight arrays.

    When events are padded up to a common ``n_samples`` with ``prior = +inf`` (so the
    padded samples carry zero weight), pass ``event_counts`` -- a length-``n_events``
    array of each event's *real* sample count -- so the Monte-Carlo mean and variance
    divide by the real count rather than the padded width. ``event_counts=None``
    reproduces the equal-count behaviour (divide by ``n_samples``).

    Returns ``(n_events, counts, numerators, square_sums)`` where
    ``numerators[i] = logsumexp(w_i) - log(c_i)`` and
    ``square_sums[i] = logsumexp(2 w_i) - 2 log(c_i)``.
    """
    if isinstance(event_weights, (list, tuple)):
        n_events = len(event_weights)
        counts = jnp.array([w.shape[0] for w in event_weights])
        numerators = jnp.stack([scs.logsumexp(w) for w in event_weights]) - jnp.log(counts)
        square_sums = jnp.stack([scs.logsumexp(2 * w) for w in event_weights]) - 2 * jnp.log(counts)
    else:
        n_events, minimum_length = event_weights.shape
        counts = minimum_length if event_counts is None else event_counts
        numerators = scs.logsumexp(event_weights, axis=1) - jnp.log(counts)
        square_sums = scs.logsumexp(2 * event_weights, axis=1) - 2 * jnp.log(counts)
    return n_events, counts, numerators, square_sums


@partial(jit, static_argnames=['rate_likelihood','return_likelihood_info'])
def hierarchical_likelihood(event_weights, denominator_weights, total_injections, live_time=1, rate_likelihood=False, return_likelihood_info=True, event_counts=None):
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
    event_counts : jnp.ndarray, optional
        Length-(n_events) array of each event's real (un-padded) sample count. If
        None, divides by n_samples (equal-count behaviour).

    Returns
    -------
    tuple
        If `return_likelihood_info=True`:
            (lnL, var, [pe_lnL, vt_lnL], [pe_var, vt_var])
        else:
            (lnL, var)
    """
    n_events, minimum_length = event_weights.shape
    counts = minimum_length if event_counts is None else event_counts
    numerators = scs.logsumexp(event_weights, axis=1) - jnp.log(counts) # means
    denominator = scs.logsumexp(denominator_weights) - jnp.log(total_injections)

    pe_ln_likelihood = jnp.sum(numerators)
    if rate_likelihood:
        vt_ln_likelihood = n_events*jnp.log(live_time) - live_time*jnp.exp(denominator)
    else:
        vt_ln_likelihood = -n_events*denominator

    ln_likelihood = pe_ln_likelihood + vt_ln_likelihood

    square_sums = scs.logsumexp(2*event_weights, axis=1) - 2*jnp.log(counts) # square_sums
    square_sum = scs.logsumexp(2*denominator_weights) - 2*jnp.log(total_injections)

    pe_ln_likelihood_variance = jnp.sum(jnp.exp(square_sums - 2*numerators) - 1/counts)
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

def rate_likelihood(event_weights, denominator_weights, total_injections, live_time=1, event_counts=None):
    """
    Poisson rate likelihood for hierarchical inference.

    Parameters
    ----------
    event_weights : jnp.ndarray
        Rectangular (n_events, n_samples) array of log[p(θ|pop)/pi(θ|PE)] weights.
        Events with fewer than n_samples real PE samples are padded with prior=+inf
        rows (zero weight); pass ``event_counts`` so the per-event Monte-Carlo mean
        and variance divide by the real count.
    denominator_weights : jnp.ndarray
        Array of log[p(θ|pop)/pi(θ|draw)] for injections.
    total_injections : int
        Number of injection samples.
    live_time : float, optional
        Observing time in years (default 1).
    event_counts : jnp.ndarray, optional
        Length-(n_events) array of each event's real (un-padded) sample count. If
        None, divides by n_samples (equal-count behaviour).

    Returns
    -------
    tuple
        (lnL, expected_events, pe_var, vt_var, total_var)
    """
    n_events, counts, numerators, square_sums = _per_event_moments(event_weights, event_counts)
    denominator = scs.logsumexp(denominator_weights) - jnp.log(total_injections)

    pe_ln_likelihood = jnp.sum(numerators)

    nexp = live_time*jnp.exp(denominator)
    vt_ln_likelihood = n_events*jnp.log(live_time) - nexp
    ln_likelihood = pe_ln_likelihood + vt_ln_likelihood

    square_sum = scs.logsumexp(2*denominator_weights) - 2*jnp.log(total_injections)

    pe_neffs = 1 / (jnp.exp(square_sums - 2*numerators) - 1/counts)
    pe_ln_likelihood_variance = jnp.sum(1 / pe_neffs)
    
    vt_neff = jnp.exp(2*denominator - square_sum)
    vt_ln_likelihood_variance = live_time**2 * (jnp.exp(square_sum) - jnp.exp(2*denominator)/total_injections)
    
    ln_likelihood_variance = pe_ln_likelihood_variance + vt_ln_likelihood_variance
    
    return {
        'log_likelihood': ln_likelihood, 
        'nexp': nexp, 
        'total_pe_lnL_variance': pe_ln_likelihood_variance, 
        'total_vt_lnL_variance': vt_ln_likelihood_variance, 
        'total_lnL_variance': ln_likelihood_variance, 
        'single_event_neffs': pe_neffs,
        'vt_neff': vt_neff,
    }


bbh_minima = {
    'log_mass_1': jnp.log(3),
    'mass_1': 3.,
    'mass_2': 3.,
    'mass_ratio': 0.,
    'log_mass_2': jnp.log(3),
    't': -1.,
    'cos_tilt': -1.,
    'cos_tilt_1': -1.,
    'cos_tilt_2': -1.,
    'a': 0.,
    'a_1': 0.,
    'a_2': 0.,
    'chi_eff': -1.,
    'chi_p': 0.,
    'redshift': 0.,
    # Use the same support for the auxiliary window parameter as for log_mass_1.
    'log_mass_1_window': jnp.log(3),
    'log_mass_2_window': jnp.log(3),
    'mass_ratio_window': 0,
}
bbh_maxima = {
    'log_mass_1': jnp.log(200),
    'mass_1': 200.,
    'mass_2': 200.,
    'mass_ratio': 1.,
    'log_mass_2': jnp.log(200),
    't': 1.,
    'cos_tilt': 1.,
    'cos_tilt_1': 1.,
    'cos_tilt_2': 1.,
    'a': 1.,
    'a_1': 1.,
    'a_2': 1.,
    'chi_eff': 1.,
    'chi_p': 1.,
    'redshift': 1.9,
    'log_mass_1_window': jnp.log(200),
    'log_mass_2_window': jnp.log(200),
    'mass_ratio_window': 1,
}

gwparameter_to_model = {
    'mass_1': BrokenPowerlawPlusTwoPeaks_PrimaryMass, #(data, slope, minimum, maximum, delta_m, mpp, sigpp, lam)
    'log_mass_1': BrokenPowerlawPlusTwoPeaks_PrimaryMass, #(data, slope, minimum, maximum, delta_m, mpp, sigpp, lam)
    'mass_ratio': SimplePowerlaw_MassRatio, #(data, slope)
    'redshift': PowerlawRedshiftPsi, #(data, lamb, maximum):
    'chi_eff': chieff_gaussian, #(data, mean, sig)
    'chi_p': chip_gaussian, #(data, mean, sig)
    'spin': spin_default, #(data, mu, var, sig, zeta)
    'a': iid_normal_spin, #(data, mu, var)
    't': tilt_iid, #(data, mu, sig, zeta)
    'log_mass_1_window': PlanckWindow_PrimaryMass, #(data, mmin, delta_m)
    'log_mass_2_window':  PlanckWindow_SecondaryMass, #(data, mmin, delta_m)
    'mass_ratio_window':  PlanckWindow_MassRatio, #(data, qmin, delta_q)
}

typical_hyperparameters = {
    'alpha':3, 'beta':2, 'mmin':2, 'mmax':199, 'delta_m':5, 'mpp':35, 'sigpp':5, 
    'lam':0.005, 'lamb':2, 'mu_x':0.06, 'sig_x':0.1, 'mu_xp':0.3, 'sig_xp':0.2, 
    'mu_spin':0.2, 'var_spin':0.1, 'mu_tilt':0.6, 'sig_tilt':0.6, 'zeta_tilt':0.5, 
    'lnsigma':-1, 'lncor': -5, 'mean': 0, 'qmin': 0.02, 'max_z': 1.9,
}

parameter_values = {
    'mass_1': 40., 'log_mass_1': np.log(40.), 'mass_ratio': 0.9, 'chi_eff': 0., 
    'chi_p': 0.3, 'redshift': 0.2, 'a_1': 0.2, 'a_2': 0.2, 'cos_tilt_1': 0.,
    'cos_tilt_2': 0.
    }

gwparameter_to_hyperparameters = {
    'mass_1': ['alpha_1', 'alpha_2', 'mmin', 'break_mass', 'delta_m_1', 'lam_fractions', 'mpp_1', 'sigpp_1', 'mpp_2', 'sigpp_2'], 
    'log_mass_1': ['alpha_1', 'alpha_2', 'mmin', 'break_mass', 'delta_m_1', 'lam_fractions', 'mpp_1', 'sigpp_1', 'mpp_2', 'sigpp_2'], 
    'log_mass_1_window': ['mmin', 'delta_m'],
    'log_mass_2_window': ['mmin', 'delta_m'], # use same for simplicity
    'mass_ratio_window': ['qmin', 'delta_q'], 
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
    'qmin': ([0.1], dist.Delta), 
    'delta_q': ([0,0.3], dist.Uniform),
    'mmin': ([3, 10], dist.Uniform), 
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
    'alpha_1': ([-4, 12], dist.Uniform),
    'alpha_2': ([-4, 12], dist.Uniform),
    'break_mass': ([20, 50], dist.Uniform),
    'delta_m_1': ([0, 10], dist.Uniform),
    'lam_fractions': ([jnp.ones(3)], dist.Dirichlet),
    'mpp_1': ([5, 20], dist.Uniform),
    'sigpp_1': ([0, 10], dist.Uniform),
    'mpp_2': ([25, 60], dist.Uniform),
    'sigpp_2': ([0, 10], dist.Uniform),
}

map_to_gwpop_parameters = {
    'mass_1': ['mass_1'],
    'log_mass_1': ['log_mass_1'],
    'mass_2': ['mass_2'],
    'log_mass_2': ['log_mass_2'],
    'mass_ratio': ['mass_ratio'],
    'redshift': ['redshift'],
    'redshift_psi': ['redshift_psi'],
    'chi_eff': ['chi_eff'],
    'chi_p': ['chi_p'],
    'a_1': ['a_1'],
    'a_2': ['a_2'],
    'cos_tilt_1': ['cos_tilt_1'],
    'cos_tilt_2': ['cos_tilt_2'],
    'spin': ['a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2'],
    'a': ['a_1', 'a_2'],
    't': ['cos_tilt_1', 'cos_tilt_2'],
    'cos_tilt': ['cos_tilt_1', 'cos_tilt_2'],
    'log_mass_1_window': ['log_mass_1'],
    'mass_ratio_window': ['mass_ratio'],
    'log_mass_2_window': ['log_mass_2'],
}
