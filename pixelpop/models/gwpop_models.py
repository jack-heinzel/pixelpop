from astropy.cosmology import Planck15
from jax import jit
import jax.numpy as jnp
import jax.scipy.special as scs
import numpy as np
from functools import partial

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
    
    return jnp.where(condition, negx_valid-jnp.log(1+jnp.exp(negx_valid)), -jnp.log(1+jnp.exp(-posx_valid)))

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

def PowerlawPlusPeak_PrimaryMass(data, alpha, minimum, maximum, delta_m, mpp, sigpp, lam):
    '''
    data is either a dictionary containing 'log_mass_1' or 'mass_1', or an jax.numpy array 
    of mass_1 samples

    slope, minimum, maximum, delta_m, mpp, sigpp, and lam are the standard PP mass parameters
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
    power_law = powerlaw(m1, slope, minimum, maximum)
    smoothed_pl = power_law + m_smoother(m1, minimum, delta_m)
    peak = gaussian(m1, mpp, sigpp)
    pm1 = jnp.logaddexp(smoothed_pl + jnp.log(1-lam), peak + jnp.log(lam))
    
    m1s_test = jnp.linspace(2.0, 100., 1000)
    dm1 = m1s_test[1] - m1s_test[0]
    power_law_test = powerlaw(m1s_test, slope, minimum, maximum)
    smoothed_pl_test = power_law_test + m_smoother(m1s_test, minimum, delta_m)
    peak_test = gaussian(m1s_test, mpp, sigpp)
    smoothed_pl_test = jnp.logaddexp(smoothed_pl_test + jnp.log(1-lam), peak_test + jnp.log(lam))
    
    pm1 -= scs.logsumexp(smoothed_pl_test) + jnp.log(dm1) # simple Riemann rule
    if isLogMass: # include jacobian
        pm1 = pm1 + data['log_mass_1']
    return pm1

@jit
def PowerlawPlusPeak_PrimaryMass_NormFirst(data, alpha, minimum, maximum, delta_m, mpp, sigpp, lam):
    '''
    data is either a dictionary containing 'log_mass_1' or 'mass_1', or an jax.numpy array 
    of mass_1 samples

    slope, minimum, maximum, delta_m, mpp, sigpp, and lam are the standard PP mass parameters

    This is NOT the standard PP model, this normalizes the smoothed powerlaw FIRST before adding the gaussian peak
    This basically changes the definition of lam, but is equivalent up to a redefinition. That is, sampling with
    this model will be identical to sampling with the conventional model.
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
    power_law = powerlaw(m1, slope, minimum, maximum)
    smoothed_pl = power_law + m_smoother(m1, minimum, delta_m)
    m1s_test = jnp.linspace(2.0, 100., 1000)
    dm1 = m1s_test[1] - m1s_test[0]
    power_law_test = powerlaw(m1s_test, slope, minimum, maximum)
    smoothed_pl_test = power_law_test + m_smoother(m1s_test, minimum, delta_m)
    smoothed_pl -= scs.logsumexp(smoothed_pl_test) + jnp.log(dm1) # simple Riemann rule
    peak = gaussian(m1, mpp, sigpp)
    pm1 = jnp.logaddexp(smoothed_pl + jnp.log(1-lam), peak + jnp.log(lam))
    if isLogMass: # include jacobian
        pm1 = pm1 + data['log_mass_1']
    return pm1

def NoSmoothedPowerlawPlusPeak_PrimaryMass(data, alpha, minimum, maximum, mpp, sigpp, lam):
    '''
    Modifies how the PP model is smoothed, instead of using a numerically difficult log_expit function

    slope, minimum, maximum, mpp, sigpp, and lam are the standard PP mass parameters
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
    power_law = powerlaw(m1, slope, minimum, maximum)
    peak = gaussian(m1, mpp, sigpp)
    pm1 = jnp.logaddexp(power_law + jnp.log(1-lam), peak + jnp.log(lam))
    if isLogMass: # include jacobian
        pm1 = pm1 + data['log_mass_1']
    return pm1

@jit
def powerlaw(data, slope, minimum, maximum):
    norm = -jnp.log(jnp.abs(slope + 1)) + jnp.log(jnp.abs(maximum**(slope+1) - minimum**(slope+1)))
    
    window = jnp.logical_and(data >= minimum, data <= maximum)
    p = jnp.where(window, slope*jnp.log(data), -100.*jnp.ones_like(data))
    return p - norm

@jit
def gaussian(data, mean, sig):
    
    px = -(data - mean)**2 / 2 / sig**2
    norm = 0.5*jnp.log(2*jnp.pi*sig**2)
    return px - norm

@jit
def chieff_gaussian(data, mean, sig):
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    return gaussian(x, mean, sig)

@jit
def trunc_gaussian(data, mean, sig, lower, upper):
    px = -(data - mean)**2 / 2 / sig**2
    up = (upper - mean) / sig / jnp.sqrt(2)
    lo = (lower - mean) / sig / jnp.sqrt(2)
    trunc = 0.5*(scs.erf(up) - scs.erf(lo))
    norm = 0.5*jnp.log(2*jnp.pi*sig**2) + jnp.log(trunc)
    return px - norm

# Sofia implements a truncated gaussian that cuts at the limits
@jit
def trunc_gaussian_2(data, mean, sig, lower, upper):
    px = -(data - mean)**2 / 2 / sig**2
    width = 0.001 #TODO: hardcoding it for now.
    taper_l = jnp.where(data > lower, 0, -((data-lower)/width)**2)
    taper_r = jnp.where(data < upper, 0, -((data-upper)/width)**2)
    px = px + taper_l + taper_r
    up = (upper - mean) / sig / jnp.sqrt(2)
    lo = (lower - mean) / sig / jnp.sqrt(2)
    trunc = 0.5*(scs.erf(up) - scs.erf(lo))
    norm = 0.5*jnp.log(2*jnp.pi*sig**2) + jnp.log(trunc)
    return px - norm

# Sofia implements a mixture of two gaussians for the chieff model
@jit
def chieff_two_gaussians(data, mean1, sig1, mean2, sig2, lamb_x):
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    gaussian_1 = gaussian(x, mean1, sig1)
    gaussian_2 = gaussian(x, mean2, sig2)
    model = jnp.logaddexp(jnp.log(lamb_x) + gaussian_1, jnp.log(1 - lamb_x) + gaussian_2)    
    return model

@jit
def chieff_two_trunc_gaussians(data, mean1, sig1, lower1, upper1, mean2, sig2, lower2, upper2, lamb_x):
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    trunc_gaussian_1 = trunc_gaussian_2(x, mean1, sig1, lower1, upper1)
    trunc_gaussian_2 = trunc_gaussian_2(x, mean2, sig2, lower2, upper2)
    model = jnp.logaddexp(jnp.log(lamb_x) + trunc_gaussian_1, jnp.log(1 - lamb_x) + trunc_gaussian_2)
    return model

@jit
def PowerlawRedshift(data, lamb, normalize=True):
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data

    zs_fixed = np.linspace(1e-5, 1.9, 1000)
    fixed_ln_dvc_dz = jnp.log(Planck15.differential_comoving_volume(zs_fixed).value)
    ln_dvc_dz = jnp.interp(z, zs_fixed, fixed_ln_dvc_dz)
    ln_p = ln_dvc_dz + (lamb - 1) * jnp.log(1. + z)
    if normalize:
        dz = zs_fixed[1] - zs_fixed[0]
        test_ln_p = fixed_ln_dvc_dz + (lamb - 1) * jnp.log(1. + zs_fixed)
        ln_norm = scs.logsumexp(test_ln_p) + jnp.log(dz)
        ln_p -= ln_norm

    window = jnp.logical_and(z >= 0., z <= 1.9)
    p = jnp.where(window, ln_p, -100.*jnp.ones_like(z))
    return p

@jit
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

@jit
def PowerlawPlusPeak(data, alpha, beta, mmin, mmax, delta_m, mpp, sigpp, lam):
    pm1 = PowerlawPlusPeak_PrimaryMass(data, alpha, mmin, mmax, delta_m, mpp, sigpp, lam)
    pq = PowerlawPlusPeak_MassRatio(data, beta, mmin, delta_m)

    return pm1 + pq

@jit
def PowerlawPlusPeak_NormFirst(data, alpha, beta, mmin, mmax, delta_m, mpp, sigpp, lam):
    pm1 = PowerlawPlusPeak_PrimaryMass_NormFirst(data, alpha, mmin, mmax, delta_m, mpp, sigpp, lam)
    pq = PowerlawPlusPeak_MassRatio(data, beta, mmin, delta_m)

    return pm1 + pq


@jit
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

@jit
def beta_spin(spin_mag, alpha, beta):
    ln_a = jnp.log(spin_mag)
    ln_1ma = jnp.log(1. - spin_mag)
    ln_p = (alpha - 1) * ln_a + (beta - 1) * ln_1ma

    norm = scs.gammaln(alpha) + scs.gammaln(beta) - scs.gammaln(alpha + beta)
    return ln_p - norm

@jit
def iid_beta_spin(data, mu, var):
    alpha, beta = mu_var_to_alpha_beta(mu, var)
    return beta_spin(data['a_1'], alpha, beta) + beta_spin(data['a_2'], alpha, beta)

@jit
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

@jit
def tilt_default(data, sig, zeta):
    '''
    Here the tilt distribution is NOT iid, either BOTH isotropic or BOTH from field, truncated gaussian
    '''
    return tilt_model(data, 1., sig, zeta)

@jit
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
    
@jit
def spin_iid(data, mu, var, mu_tilt, sig_tilt, zeta):
    return iid_beta_spin(data, mu, var) + tilt_iid(data, mu_tilt, sig_tilt, zeta)

@jit
def spin_default(data, mu, var, sig_tilt, zeta):
    return iid_beta_spin(data, mu, var) + tilt_default(data, sig_tilt, zeta)

@partial(jit, static_argnames=['rate_likelihood'])
def hierarchical_likelihood(event_weights, denominator_weights, total_injections, live_time=1, rate_likelihood=False, return_likelihood_info=True):
    '''
    event weights are a n_events by minimum_length 2d array of ln[p(theta | lambda) / prior(theta)]
    denominator weights are a 1d array of p(theta|lambda) / prior(theta)
    '''
    n_events, minimum_length = event_weights.shape
    numerators = scs.logsumexp(event_weights, axis=1) - jnp.log(minimum_length) # means
    denominator = scs.logsumexp(denominator_weights) - jnp.log(total_injections)

    # Sofia (05/01/24): compute pe_ln_likelihood, vt_ln_likelihood
    pe_ln_likelihood = jnp.sum(numerators)
    if rate_likelihood:
        vt_ln_likelihood = n_events*jnp.log(live_time) - live_time*jnp.exp(denominator)
    else:
        vt_ln_likelihood = -n_events*denominator
    ln_likelihoods = [pe_ln_likelihood, vt_ln_likelihood]
    ln_likelihood = pe_ln_likelihood + vt_ln_likelihood
    square_sums = scs.logsumexp(2*event_weights, axis=1) - 2*jnp.log(minimum_length) # square_sums
    square_sum = scs.logsumexp(2*denominator_weights) - 2*jnp.log(total_injections)
    
    # Sofia (05/01/24): compute pe_ln_likelihood_variance, vt_ln_likelihood_variance
    pe_ln_likelihood_variance = jnp.sum(jnp.exp(square_sums - 2*numerators) - 1/minimum_length)
    # print(ln_likelihood_variance)
    if rate_likelihood:
        vt_ln_likelihood_variance = live_time**2 * (jnp.exp(square_sum) - jnp.exp(2*denominator)/total_injections)
    else:
        vt_ln_likelihood_variance += n_events**2 * (jnp.exp(square_sum - 2*denominator) - 1/total_injections)
    ln_likelihood_variance = pe_ln_likelihood_variance + vt_ln_likelihood_variance
    ln_likelihood_variances = [pe_ln_likelihood_variance, vt_ln_likelihood_variance]
    if return_likelihood_info:
        return ln_likelihood, ln_likelihood_variance, ln_likelihoods, ln_likelihood_variances
    else:
        return ln_likelihood, ln_likelihood_variance