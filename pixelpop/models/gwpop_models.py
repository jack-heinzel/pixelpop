import wcosmo
from astropy import units
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
    if (x < 0.0) {
        return x - std::log1p(std::exp(x));
    }
    else {
        return -std::log1p(std::exp(-x));
    }
    exact same expression, just more numerically stable for each side of 0
    log(1 / [1 + exp(-x)]) = log(exp(x) / (exp(x) + 1)) = x - log(1 + exp(x))

    exactly as done in scipy.special.log_expit

    DANGEROUS: https://github.com/google/jax/issues/1052 gradients will be nan.

    cannot use a single `where', must use the "double-where" trick

    consider differentiating this function for x = -1000, using naive single-where method
    the arguments of where are (True, -1000, jnp.inf). This causes trouble for gradients as the 
    divergence is propagated along, eventually it is multiplied by 0 to remove it, but 0*inf is an issue.
    With the double where method, now evaulating the function for x=-1000 gives you (True, -1000, 0.)
    and this works because the forward differentiation is done serially, so we never get an inf
    """
    condition = x < 0
    posx_valid = jnp.where(condition, 0, x) # in forward differentiation, gradient is 0 for condition, 1 where false
    negx_valid = jnp.where(condition, x, 0) # in forward differentiation, gradient is 0 for condition, 1 where false
    
    return jnp.where(condition, negx_valid-jnp.log1p(jnp.exp(negx_valid)), -jnp.log1p(jnp.exp(-posx_valid)))

def m_smoother(m1s, minimum, delta, buffer=1e-3):
    '''
    remember, logspace
    return log(1) if greater than minimum + delta
    return log(0) if less than minimum
    return log(1 / 1+f(m-mmin, delta)) if inside minimum and minimum + delta

    standard powerlaw + peak smoother: https://arxiv.org/pdf/2111.03634.pdf B5
    '''

    m_prime = jnp.clip(m1s - minimum, buffer, delta-buffer)
    return log_expit(-delta/m_prime - delta/(m_prime - delta))

def powerlaw(data, slope, minimum, maximum):
    norm = jnp.where(
        jnp.isclose(slope, -1), 
        jnp.log(jnp.log(maximum / minimum)),
        -jnp.log(jnp.abs(slope + 1)) + jnp.log(jnp.abs(maximum**(slope+1) - minimum**(slope+1)))
    )
    window = jnp.logical_and(data >= minimum, data <= maximum)
    p = jnp.where(window, slope*jnp.log(data), -INF*jnp.ones_like(data))
    return p - norm

def gaussian(data, mean, sig):
    px = -(data - mean)**2 / 2 / sig**2
    norm = 0.5*jnp.log(2*jnp.pi*sig**2)
    return px - norm

def PowerlawPlusPeak_PrimaryMass(data, alpha, minimum, maximum, delta_m, mpp, sigpp, lam):
    '''
    Parameters
    ----------
    data: dict or jax.numpy.ndarray
        Either a dictionary containing 'log_mass_1' or 'mass_1' OR a jax.numpy.ndarray
        containing the mass_1 samples
    alpha: float
        -alpha is the slope of the powerlaw index in the powerlaw continuum
    minimum: float
        minimum BH mass
    maximum: float
        maximum BH mass
    delta_m: float
        smoothing length, between minimum and minimum+delta_m
    mpp: float
        location parameter of the Gaussian peak
    sigpp: float
        width parameter of the Gaussian peak
    lam: float
        the fraction of sources in the Gaussian peak
    
    Returns
    -------
    pm1: jax.numpy.ndarray
        array of shape m1 of ln probability densities
    '''
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
    GWTC-4.0 population default mass model, 
    NOTE: lam_fractions should be a tuple of length 3

    Parameters
    ----------
    data: dict or jax.numpy.ndarray
        Either a dictionary containing 'log_mass_1' or 'mass_1' OR a jax.numpy.ndarray
        containing the mass_1 samples
    alpha_1: float
        -alpha_1 is the slope of the powerlaw index in the powerlaw continuum below 
        break_mass (m_1 < break_mass)
    alpha_2: float
        -alpha_2 is the slope of the powerlaw index in the powerlaw continuum above 
        break_mass (m_1 > break_mass)
    mmin: float
        minimum (primary) BH mass
    break_mass: float
        mass at which powerlaw continuum transitions from slope -alpha_1 to -alpha_2
    delta_m_1: float
        smoothing length for primary mass, between mmin and mmin+delta_m_1
    lam_fractions: tuple of length 3
        lam_fractions[0] is the fraction of sources in the powerlaw continuum,
        lam_fractions[1] is the fraction of sources in the low-mass Gaussian component,
        lam_fractions[2] is the fraction of sources in the high-mass Gaussian component. 
        These are sampled from a distribution with support on the 3-simplex, so 
        lam_fractions[0] + lam_fractions[1] + lam_fractions[2] = 1
    mpp_1: float
        location parameter of the low-mass Gaussian peak
    sigpp_1: float
        width parameter of the low-mass Gaussian peak
    mpp_2: float
        location parameter of the high-mass Gaussian peak
    sigpp_2: float
        width parameter of the high-mass Gaussian peak
    mmax: float
        maximum BH mass
    gaussian_mass_maximum:
        mass at which to truncate the gaussians.
    
    Returns
    -------
    pm1: jax.numpy.ndarray
        array of shape m1 of ln probability densities
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
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    return gaussian(x, mean, sig)

def trunc_gaussian(data, mean, sig, lower, upper):
    in_support = jnp.logical_and(data < upper, data > lower)
    px = -(data - mean)**2 / 2 / sig**2
    up = (upper - mean) / sig / jnp.sqrt(2)
    lo = (lower - mean) / sig / jnp.sqrt(2)
    trunc = 0.5*(scs.erf(up) - scs.erf(lo))
    norm = 0.5*jnp.log(2*jnp.pi*sig**2) + jnp.log(trunc)
    return jnp.where(in_support, px - norm, -jnp.inf*jnp.ones_like(data))

def lognormal(data, mean, sig):
    px = -(jnp.log(data) - mean)**2 / 2 / sig**2
    denom = jnp.log(data*sig*jnp.sqrt(2*jnp.pi))
    return px - denom

def PowerlawRedshift(data, lamb, max_z=1.9, normalize=True, return_normalization=False):
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    zs_fixed = np.linspace(1e-5, max_z, 1000)
    fixed_ln_dvc_dz = jnp.log(
        4*jnp.pi*COSMO.differential_comoving_volume(zs_fixed).to(units.Gpc**3 / units.sr).value
        )
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
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    ln_p = lamb * jnp.log(1. + z)

    window = jnp.logical_and(z >= 0., z <= max_z)
    p = jnp.where(window, ln_p, -INF*jnp.ones_like(z))
    return p


def MadauDickinsonRedshift(data, gamma, kappa, z_peak, z_max=1.9, normalize=True, return_normalization=False):
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    zs_fixed = np.linspace(1e-5, z_max, 1000)
    fixed_ln_dvc_dz = jnp.log(
        4*jnp.pi*COSMO.differential_comoving_volume(zs_fixed).to(units.Gpc**3 / units.sr).value
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
    '''
    data is either a dictionary containing 'log_mass_1' or 'mass_1', or an jax.numpy array 
    of mass_1 samples

    slope, minimum, maximum, delta_m, mpp, sigpp, and lam are the standard PP mass parameters
    '''
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
    '''
    Simple powerlaw function for fitting mass ratio distribution, using a global
    minimum BH mass
    '''
    try:
        m1 = jnp.exp(data['log_mass_1'])
    except KeyError:
        m1 = data['mass_1']
    q = data['mass_ratio']

    power_law = powerlaw(q, slope, minimum/m1, jnp.ones_like(m1))
    return power_law

def SimplePowerlaw_MassRatio(data, slope, qmin):
    '''
    Simple function for fitting mass ratio distribution, using no global minimum
    BH mass, can be as low as 0.02
    '''
    q = data['mass_ratio']

    power_law = powerlaw(q, slope, qmin, 1.)
    return power_law


def PowerlawPlusPeak(data, alpha, beta, mmin, mmax, delta_m, mpp, sigpp, lam):
    pm1 = PowerlawPlusPeak_PrimaryMass(data, alpha, mmin, mmax, delta_m, mpp, sigpp, lam)
    pq = PowerlawPlusPeak_MassRatio(data, beta, mmin, delta_m)

    return pm1 + pq

def smooth(x, cutoff, width):
    # less than cutoff return 1
    # first derivative is continuous, so all good for autodiff
    # interestingly cannot do this as a if statement...
    return jnp.where(x<cutoff, 0., -((x-cutoff)/width)**2)

def mu_var_to_alpha_beta(mu, var):
    nu = (mu*(1-mu)/var) - 1
    alpha = mu * nu
    beta = (1-mu) * nu
    return alpha, beta

def beta_spin_mv(spin_mag, mu, var):
    alpha, beta = mu_var_to_alpha_beta(mu, var)
    return beta_spin(spin_mag, alpha, beta)


def beta_spin(spin_mag, alpha, beta):
    ln_a = jnp.log(spin_mag)
    ln_1ma = jnp.log(1. - spin_mag)
    ln_p = (alpha - 1) * ln_a + (beta - 1) * ln_1ma

    norm = scs.gammaln(alpha) + scs.gammaln(beta) - scs.gammaln(alpha + beta)
    return ln_p - norm


def iid_beta_spin(data, mu, var):
    alpha, beta = mu_var_to_alpha_beta(mu, var)
    return beta_spin(data['a_1'], alpha, beta) + beta_spin(data['a_2'], alpha, beta)

def iid_normal_spin(data, mu, var):
    sig = jnp.sqrt(var)
    return trunc_gaussian(data['a_1'], mu, sig, 0, 1) + trunc_gaussian(data['a_2'], mu, sig, 0, 1)

def iid_normal_spin_fms(data, mu, var):
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
            regions[f'mass_{ii}'] < 2.5, 
            trunc_gaussian(data[f'a_{ii}'], mu, sig, 0, 0.4), 
            trunc_gaussian(data[f'a_{ii}'], mu, sig, 0, 1)
            )
        total_prob += probs
    
    return total_prob

def tilt_model(data, mu, sig, zeta):
    '''
    Only difference here is that mu is allowed to be != 1, a free parameter
    '''
    pfield1 = trunc_gaussian(data['cos_tilt_1'], mu, sig, -1, 1)
    pfield2 = trunc_gaussian(data['cos_tilt_2'], mu, sig, -1, 1)

    pisotropic = jnp.log(jnp.ones_like(data['cos_tilt_1']) / 4)
    pfield = pfield1 + pfield2

    ln_zeta = jnp.log(zeta)
    ln_1mzeta = jnp.log(1 - zeta)

    return jnp.logaddexp(ln_zeta + pfield, ln_1mzeta + pisotropic)


def tilt_default(data, sig, zeta):
    '''
    Here the tilt distribution is NOT iid, either BOTH isotropic or BOTH from field, truncated gaussian
    '''
    return tilt_model(data, 1., sig, zeta)


def tilt_iid(data, mu, sig, zeta):
    '''
    here, the tilt distribution is assumed to be IID, a truncated normal plus a isotropic component
    '''
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
    '''
    event weights are a n_events by minimum_length 2d array of ln[p(theta | lambda) / prior(theta)]
    denominator weights are a 1d array of p(theta|lambda) / prior(theta)
    '''
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
    '''
    event weights are a n_events by minimum_length 2d array of ln[p(theta | lambda) / prior(theta)]
    denominator weights are a 1d array of p(theta|lambda) / prior(theta)
    '''
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


bbh_minima = {'log_mass_1': jnp.log(3), 'mass_1': 3., 'mass_2': 3., 'mass_ratio': 0., 'log_mass_2': jnp.log(3), 'chi_eff': -1., 'redshift': 0.}
bbh_maxima = {'log_mass_1': jnp.log(200), 'mass_1': 200., 'mass_2': 200., 'mass_ratio': 1., 'log_mass_2': jnp.log(200), 'chi_eff': 1., 'redshift': 2.4}

gwparameter_to_model = {
    'mass_1': PowerlawPlusPeak_PrimaryMass, #(data, slope, minimum, maximum, delta_m, mpp, sigpp, lam)
    'log_mass_1': PowerlawPlusPeak_PrimaryMass, #(data, slope, minimum, maximum, delta_m, mpp, sigpp, lam)
    'mass_ratio': SimplePowerlaw_MassRatio, #(data, slope)
    'redshift': PowerlawRedshift, #(data, lamb, minimum, maximum, normalize=False):
    'redshift_psi': PowerlawRedshiftPsi, #(data, lamb, minimum, maximum, normalize=False):
    'chi_eff': chieff_gaussian, #(data, mean, sig)
    'spin': spin_default, #(data, mu, var, sig, zeta)
    'a': iid_normal_spin, #(data, mu, var)
    't': tilt_iid, #(data, mu, sig, zeta)
}

typical_hyperparameters = {
    'alpha':3, 'beta':2, 'mmin':2, 'mmax':199, 'delta_m':5, 'mpp':35, 'sigpp':5, 
    'lam':0.005, 'lamb':2, 'mu_x':0.06, 'sig_x':0.1, 'mu_spin':0.2, 'var_spin':0.1, 
    'mu_tilt':0.6, 'sig_tilt':0.6, 'zeta_tilt':0.5, 'lnsigma':-1, 'lncor': -5, 
    'mean': 0, 'qmin': 0.02, 'max_z': 2.4,
}

parameter_values = {
    'mass_1': 40., 'log_mass_1': np.log(40.), 'mass_ratio': 0.9, 'chi_eff': 0., 'redshift': 0.2, 
    'a_1': 0.2, 'a_2': 0.2, 'cos_tilt_1': 0., 'cos_tilt_2': 0.
    }

gwparameter_to_hyperparameters = {
    'mass_1': ['alpha', 'mmin', 'mmax', 'delta_m', 'mpp', 'sigpp', 'lam'], 
    'log_mass_1': ['alpha', 'mmin', 'mmax', 'delta_m', 'mpp', 'sigpp', 'lam'], 
    'mass_ratio': ['beta', 'qmin'], 
    'redshift': ['lamb', 'max_z'],
    'redshift_psi': ['lamb', 'max_z'],
    'chi_eff': ['mu_x', 'sig_x'], 
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
    'mu_spin': ([0, 1], dist.Uniform),
    'var_spin': ([0.005, 0.25], dist.Uniform), 
    'mu_tilt': ([-1, 1], dist.Uniform), 
    'sig_tilt': ([0.1, 4], dist.Uniform), 
    'zeta_tilt': ([0, 1], dist.Uniform), 
    'z_minimum': ([0.], dist.Delta), 
    'max_z': ([2.4], dist.Delta),
}

map_to_gwpop_parameters = {
    'mass_1': ['mass_1'], 'log_mass_1': ['log_mass_1'], 'mass_2': ['mass_2'], 'log_mass_2': ['log_mass_2'], 
    'mass_ratio': ['mass_ratio'], 'redshift': ['redshift'], 'redshift_psi': ['redshift_psi'], 'chi_eff': ['chi_eff'], 
    'a_1': ['a_1'], 'a_2': ['a_2'], 'cos_tilt_1': ['cos_tilt_1'], 'cos_tilt_2': ['cos_tilt_2'], 
    'spin': ['a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2'], 'a': ['a_1', 'a_2'], 't': ['cos_tilt_1', 'cos_tilt_2'],
}