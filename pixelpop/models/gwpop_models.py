from astropy.cosmology import Planck15
from astropy import units
from jax import jit, lax
import numpyro
import jax.numpy as jnp
import jax.scipy.special as scs
import numpy as np
from functools import partial
from jax.scipy.special import erf
from jax.scipy.special import gamma

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
    m1s_test = jnp.linspace(2.0, 200., 2000)
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


def powerlaw(data, slope, minimum, maximum):
    norm = -jnp.log(jnp.abs(slope + 1)) + jnp.log(jnp.abs(maximum**(slope+1) - minimum**(slope+1)))
    
    window = jnp.logical_and(data >= minimum, data <= maximum)
    p = jnp.where(window, slope*jnp.log(data), -100.*jnp.ones_like(data))
    return p - norm


def gaussian(data, mean, sig):
    
    px = -(data - mean)**2 / 2 / sig**2
    norm = 0.5*jnp.log(2*jnp.pi*sig**2)
    return px - norm


def chieff_gaussian(data, mean, sig):
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    return gaussian(x, mean, sig)


def trunc_gaussian(data, mean, sig, lower, upper):
    px = -(data - mean)**2 / 2 / sig**2
    up = (upper - mean) / sig / jnp.sqrt(2)
    lo = (lower - mean) / sig / jnp.sqrt(2)
    trunc = 0.5*(scs.erf(up) - scs.erf(lo))
    norm = 0.5*jnp.log(2*jnp.pi*sig**2) + jnp.log(trunc)
    return px - norm

# Sofia implements a truncated gaussian that cuts at the limits

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


def ln_chieff(data, mean, sig):
    px = -(jnp.log(data) - mean)**2 / 2 / sig**2
    denom = jnp.log(data*sig*jnp.sqrt(2*jnp.pi))
    return px - denom

# Sofia implements a mixture of two gaussians for the chieff model

def chieff_two_gaussians(data, mean1, sig1, mean2, sig2, lamb_x):
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    gaussian_1 = gaussian(x, mean1, sig1)
    gaussian_2 = gaussian(x, mean2, sig2)
    model = jnp.logaddexp(jnp.log(lamb_x) + gaussian_1, jnp.log(1 - lamb_x) + gaussian_2)    
    return model


def chieff_two_trunc_gaussians(data, mean1, sig1, lower1, upper1, mean2, sig2, lower2, upper2, lamb_x):
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    trunc_gaussian_1 = trunc_gaussian_lims(x, mean1, sig1, lower1, upper1)
    trunc_gaussian_2 = trunc_gaussian_lims(x, mean2, sig2, lower2, upper2)
    model = jnp.logaddexp(jnp.log(lamb_x) + trunc_gaussian_1, jnp.log(1 - lamb_x) + trunc_gaussian_2)
    return model


def log_tukey(x, tx0, tr, tk, eps=1e-10, normalize=True):
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


def chieff_gaussian_tukey(data, mean, sig, tx0, tr, tk, lamb_x):
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    gaussian = trunc_gaussian_lims(x, mean, sig, lower=-1, upper=-1)
    tukey = log_tukey(x, tx0, tr, tk)
    model = jnp.logaddexp(jnp.log(lamb_x) + gaussian, jnp.log(1 - lamb_x) + tukey)
    return model


def chieff_lognormal_tukey(data, mu_x, sig_x, tx0_x, tr_x, tk_x, lamb_x):
    if isinstance(data, dict):
        x = data['chi_eff']
    else:
        x = data
    gaussian = ln_chieff(x, mu_x, sig_x)
    tukey = log_tukey(x, tx0_x, tr_x, tk_x)
    model = jnp.logaddexp(jnp.log(lamb_x) + gaussian, jnp.log(1 - lamb_x) + tukey)
    return model


def PowerlawRedshift(data, lamb, max_z=1.9, normalize=True, return_normalization=False):
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    zs_fixed = np.linspace(1e-5, max_z, 1000)
    fixed_ln_dvc_dz = jnp.log(
        4*jnp.pi*Planck15.differential_comoving_volume(zs_fixed).to(units.Gpc**3 / units.sr).value
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
    p = jnp.where(window, ln_p, -100.*jnp.ones_like(z))
    return p


def PowerlawRedshiftPsi(data, lamb, max_z=1.9):
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    ln_p = lamb * jnp.log(1. + z)

    window = jnp.logical_and(z >= 0., z <= max_z)
    p = jnp.where(window, ln_p, -100.*jnp.ones_like(z))
    return p


def MadauDickinsonRedshift(data, gamma, kappa, z_peak, z_max=1.9, normalize=True):
    if isinstance(data, dict):
        z = data['redshift']
    else:
        z = data
    zs_fixed = np.linspace(1e-5, z_max, 1000)
    fixed_ln_dvc_dz = jnp.log(Planck15.differential_comoving_volume(zs_fixed).value)
    ln_dvc_dz = jnp.interp(z, zs_fixed, fixed_ln_dvc_dz)
    ln_p = ln_dvc_dz + (gamma - 1)* jnp.log(1. + z) - jnp.log(1 + ((1 + z)/(1 + z_peak))**kappa)
    if normalize:
        dz = zs_fixed[1] - zs_fixed[0]
        test_ln_p = fixed_ln_dvc_dz + (gamma - 1)* jnp.log(1. + zs_fixed) - jnp.log(1 + ((1 + zs_fixed)/(1 + z_peak))**kappa)
        # Calculate the normalization
        ln_norm = scs.logsumexp(test_ln_p) + jnp.log(dz)
        # Divide in logspace (=subtract) by normalization
        ln_p -= ln_norm
    window = jnp.logical_and(z >= 0., z <= z_max)
    p = jnp.where(window, ln_p, -100.*jnp.ones_like(z))
    return p


def skew_norm(data, mu_0, mu_1, sigma_0, sigma_1, alpha_0, alpha_1, z_max=1.9, normalize=True):
    redshift = data['redshift']
    chi_eff = data['chi_eff']
    
    mu = mu_0 + chi_eff * mu_1
    sigma = sigma_0 + chi_eff * sigma_1
    alpha = alpha_0 + chi_eff * alpha_1
    
    zs_fixed = np.linspace(1e-5, z_max, 1000)
    fixed_ln_dvc_dz = jnp.log(Planck15.differential_comoving_volume(zs_fixed).value)
    ln_dvc_dz = jnp.interp(redshift, zs_fixed, fixed_ln_dvc_dz)
    cdf = jnp.log(1 + erf(alpha * (redshift - mu) / (sigma * jnp.sqrt(2))))
    exp = -(redshift - mu) ** 2 / (2 * sigma ** 2)
    norm = jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
    result = ln_dvc_dz - jnp.log(1. + redshift) + cdf + exp - norm
    
    if normalize:
        # Define fixed values of chi_eff for normalization
        chi_eff_fixed = np.linspace(-1, 1, 1000)
        dz = zs_fixed[1] - zs_fixed[0]
        
        # Precompute variables for chi_eff_fixed
        mu_fixed = mu_0 + chi_eff_fixed * mu_1
        sigma_fixed = sigma_0 + chi_eff_fixed * sigma_1
        alpha_fixed = alpha_0 + chi_eff_fixed * alpha_1
        
        zs_fixed_ = zs_fixed[:, None]
        fixed_ln_dvc_dz = jnp.log(Planck15.differential_comoving_volume(zs_fixed_).value)
        mu_ = mu_fixed[None, ...]
        sigma_ = sigma_fixed[None, ...]
        alpha_ = alpha_fixed[None, ...]
        
        cdf_t = jnp.log(1 + erf(alpha_ * (zs_fixed_ - mu_) / (sigma_ * jnp.sqrt(2))))
        exp_t = -(zs_fixed_ - mu_) ** 2 / (2 * sigma_ ** 2)
        result_t = fixed_ln_dvc_dz - jnp.log(1. + zs_fixed_) + cdf_t + exp_t - jnp.log(sigma_ * jnp.sqrt(2 * jnp.pi))
        ln_norm_fixed = scs.logsumexp(result_t, axis=0) + jnp.log(dz)
        
        # Interpolate normalization to original chi_eff
        ln_norm = jnp.interp(chi_eff, chi_eff_fixed, ln_norm_fixed)
    
    ln_p = result - ln_norm
    conditions = jnp.stack([redshift >= 0., 
                                        redshift <= z_max,
                                        mu >= 0, mu <= 1.9, sigma > 0,
                                        np.isfinite(ln_p)], axis=0)
    window = jnp.all(conditions, axis=0)
    p = jnp.where(window, ln_p, -100. * jnp.ones_like(redshift))
    return p


def jhonson_su(data, gamma_0, gamma_1, xi_0, xi_1, delta_0, delta_1, lambda_0, lambda_1, z_max=1.9, normalize=True):
    redshift = data['redshift']
    chi_eff = data['chi_eff']
    
    gamma = gamma_0 + chi_eff*gamma_1
    xi = xi_0 + chi_eff*xi_1
    delta = delta_0 + chi_eff*delta_1
    lambd = lambda_0 + chi_eff*lambda_1
    
    zs_fixed = np.linspace(1e-5, z_max, 1000)
    fixed_ln_dvc_dz = jnp.log(Planck15.differential_comoving_volume(zs_fixed).value)
    ln_dvc_dz = jnp.interp(redshift, zs_fixed, fixed_ln_dvc_dz)
    
    f = jnp.log(delta) - jnp.log(lambd * jnp.sqrt(2*jnp.pi))
    s = -0.5*jnp.log(1 + ((redshift - xi)/lambd)**2)
    t = -0.5*(gamma + delta*np.arcsinh((redshift - xi)/lambd))**2
    result =ln_dvc_dz - jnp.log(1. + redshift) + f + s + t
    
    if normalize:
        # Define fixed values of chi_eff for normalization
        chi_eff_fixed = np.linspace(-1, 1, 1000)
        dz = zs_fixed[1] - zs_fixed[0]
        
        # Precompute variables for chi_eff_fixed
        gamma_fixed = gamma_0 + chi_eff_fixed*gamma_1
        xi_fixed = xi_0 + chi_eff_fixed*xi_1
        delta_fixed = delta_0 + chi_eff_fixed*delta_1
        lambd_fixed = lambda_0 + chi_eff_fixed*lambda_1

        zs_fixed_ = zs_fixed[:,None]
        fixed_ln_dvc_dz = jnp.log(Planck15.differential_comoving_volume(zs_fixed_).value)
        gamma_ = gamma_fixed[None,...]
        xi_ = xi_fixed[None,...]
        delta_ = delta_fixed[None,...]
        lambd_ = lambd_fixed[None, ...]
        
        f_fixed = jnp.log(delta_) - jnp.log(lambd_ * jnp.sqrt(2*jnp.pi))
        s_fixed = -0.5*jnp.log(1 + ((zs_fixed_ - xi_)/lambd_)**2)
        t_fixed = -0.5*(gamma_ + delta_*np.arcsinh((zs_fixed_ - xi_)/lambd_))**2
        result_fixed = fixed_ln_dvc_dz - jnp.log(1. + zs_fixed_) + f_fixed + s_fixed + t_fixed
        ln_norm_fixed = scs.logsumexp(result_fixed, axis=0) + jnp.log(dz)
        
        # Interpolate normalization to original chi_eff
        ln_norm = jnp.interp(chi_eff, chi_eff_fixed, ln_norm_fixed)
        
    ln_p = result - ln_norm
    conditions = jnp.stack([redshift >= 0., 
                                        redshift <= z_max,
                                        delta > 0, lambd > 0,
                                        np.isfinite(ln_p)], axis=0)
    window = jnp.all(conditions, axis=0)
    p = jnp.where(window, ln_p, -100.*jnp.ones_like(redshift))
    return p


def PERT(data, a, b_0, b_1, c_0, c_1, z_max=1.9, normalize=True):
    redshift = data['redshift']
    chi_eff = data['chi_eff']
    
    b = b_0 + chi_eff * b_1
    c = c_0 + chi_eff * c_1

    alpha = 1 + 4 * ((b - a) / (c - a))
    beta = 1 + 4 * ((c - b) / (c - a))
    gamma_factor = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    
    zs_fixed = np.linspace(1e-5, z_max, 1000)
    fixed_ln_dvc_dz = jnp.log(Planck15.differential_comoving_volume(zs_fixed).value)
    ln_dvc_dz = jnp.interp(redshift, zs_fixed, fixed_ln_dvc_dz)
    
    pos = (alpha - 1) * jnp.log(redshift - a) + (beta - 1) * jnp.log(c - redshift)
    neg = jnp.log(gamma_factor) + (alpha + beta - 1) * jnp.log(c - a)
    result = ln_dvc_dz - jnp.log(1. + redshift) + pos - neg
    
    if normalize:
        # Define fixed values of chi_eff for normalization
        chi_eff_fixed = np.linspace(-1, 1, 1000)
        dz = zs_fixed[1] - zs_fixed[0]
        
        # Precompute variables for chi_eff_fixed
        b_fixed = b_0 + chi_eff_fixed * b_1
        c_fixed = c_0 + chi_eff_fixed * c_1
        alpha_fixed = 1 + 4 * ((b_fixed - a) / (c_fixed - a))
        beta_fixed = 1 + 4 * ((c_fixed - b_fixed) / (c_fixed - a))
        gamma_factor_fixed = gamma(alpha_fixed) * gamma(beta_fixed) / gamma(alpha_fixed + beta_fixed)
        
        zs_fixed_ = zs_fixed[:, None]
        fixed_ln_dvc_dz = jnp.log(Planck15.differential_comoving_volume(zs_fixed_).value)
        a_ = a
        b_ = b_fixed[None, ...]
        c_ = c_fixed[None, ...]
        alpha_ = alpha_fixed[None, ...]
        beta_ = beta_fixed[None, ...]
        gamma_factor_ = gamma_factor_fixed[None, ...]
        
        pos_t = (alpha_ - 1) * jnp.log(zs_fixed_ - a_) + (beta_ - 1) * jnp.log(c_ - zs_fixed_)
        neg_t = jnp.log(gamma_factor_) + (alpha_ + beta_ - 1) * jnp.log(c_ - a_)
        result_t = fixed_ln_dvc_dz - jnp.log(1. + zs_fixed_) + pos_t - neg_t
        ln_norm_fixed = scs.logsumexp(result_t, axis=0) + jnp.log(dz)
        
        # Interpolate normalization to original chi_eff
        ln_norm = jnp.interp(chi_eff, chi_eff_fixed, ln_norm_fixed)
    
    ln_p = result - ln_norm
    conditions = jnp.stack([redshift >= 0., 
                            redshift <= z_max,
                            b > a, c > b,
                            np.isfinite(ln_p)], axis=0)
    window = jnp.all(conditions, axis=0)
    p = jnp.where(window, ln_p, -100. * jnp.ones_like(redshift))
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


def PowerlawPlusPeak_NormFirst(data, alpha, beta, mmin, mmax, delta_m, mpp, sigpp, lam):
    pm1 = PowerlawPlusPeak_PrimaryMass_NormFirst(data, alpha, mmin, mmax, delta_m, mpp, sigpp, lam)
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
    prob = jnp.zeros_like(data['a_1'])
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
        prob += probs
    
    return probs

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