import pytest
import numpy as np
import jax
import jax.numpy as jnp
import scipy.special as scs
import scipy.stats as stats

# If testing against gwpopulation for powerlaw/smoothing:
from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.utils import powerlaw as gwpop_powerlaw
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import (
    iid_spin_orientation_gaussian_isotropic,
    iid_spin_magnitude_beta,
    gaussian_chi_eff,
    gaussian_chi_p,
)

from .o4_default_mass_model import TwoPeakBrokenPowerLawSmoothedMassDistribution
from .o4_default_spin_model import iid_spin_magnitude_gaussian
# Assuming your models are saved in a file named `gwpop_models.py`
from pixelpop.models.gwpop_models import (
    log_expit, gaussian, trunc_gaussian, powerlaw, # generic functions
    PowerlawPlusPeak_PrimaryMass, BrokenPowerlawPlusTwoPeaks_PrimaryMass, # mass functions
    PowerlawRedshiftPsi, # redshift
    beta_spin, chieff_gaussian,  chip_gaussian, iid_beta_spin, iid_normal_spin, # spins + effective spins
    tilt_model, # tilt models
)
# import wcosmo

# Planck15_LAL = wcosmo.FlatLambdaCDM(H0=67.90, Om0=0.3065, name="Planck15_LAL")

def comp_powerlaw(data, slope, minimum, maximum):
    """convert gwpopulation to log PDF to match the PixelPop output."""
    # gwpopulation signature: powerlaw(dataset, alpha, high, low)
    return gwpop_powerlaw(data, slope, maximum, minimum)

def comp_truncnorm(data, mean, sig, lower, upper):
    """Wrapper to map standard bounds to scipy's truncnorm shape parameters."""
    a, b = (lower - mean) / sig, (upper - mean) / sig
    return stats.truncnorm.pdf(data, a=a, b=b, loc=mean, scale=sig)

def comp_powerlawpeak(data, alpha, minimum, maximum, delta_m, mpp, sigpp, lam):
    mmodel = SinglePeakSmoothedMassDistribution()
    return mmodel.p_m1(data, alpha=alpha, mmin=minimum, mmax=maximum, lam=lam, mpp=mpp, sigpp=sigpp, delta_m=delta_m)

def comp_bpl2p(data, alpha_1, alpha_2, mmin, mmax, break_mass, delta_m_1, lam_fractions, mpp_1, sigpp_1, mpp_2, sigpp_2):
    mmodel = TwoPeakBrokenPowerLawSmoothedMassDistribution(mmin=3, mmax=300, normalization_shape=(2000,500), spacing='linear')
    lam_0, lam_1, lam_2 = lam_fractions
    return mmodel.p_m1(
        data, mmin=mmin, delta_m=delta_m_1, alpha_1=alpha_1, alpha_2=alpha_2, mmax=mmax, break_mass=break_mass, 
        lam_0=lam_0, lam_1=lam_1, mpp_1=mpp_1, sigpp_1=sigpp_1, mpp_2=mpp_2, sigpp_2=sigpp_2,
        )

def comp_powerlawredshiftpsi(data, lamb):
    zmodel = PowerLawRedshift(z_max=1.9)# , cosmo_model="FlatLambdaCDM")
    return zmodel.psi_of_z(data['redshift'], lamb=lamb)

def comp_spins(data, mu, var):
    return iid_spin_magnitude_gaussian(data, mu, np.sqrt(var), 1)
    
def comp_spins_beta(data, mu, var):
    converted, _ = convert_to_beta_parameters({'mu_chi': mu, 'sigma_chi': var, 'amax': 1})
    return iid_spin_magnitude_beta(data, amax=1, alpha_chi=converted['alpha_chi'], beta_chi=converted['beta_chi'])
    
comp_tilts = lambda d, mu, sig, zeta: iid_spin_orientation_gaussian_isotropic(d, xi_spin=zeta, sigma_spin=sig, mu_spin=mu)
comp_gaussian_chieff = lambda d, mean, sig: gaussian_chi_eff(d, mu_chi_eff=mean, sigma_chi_eff=sig)
comp_gaussian_chip = lambda d, mean, sig: gaussian_chi_p(d, mu_chi_p=mean, sigma_chi_p=sig)

# Dictionary structure: 
# "model_name": (custom_jax_function, standard_ref_function, dictionary_of_kwargs)
TEST_MODELS = {
    "log_expit": (
        log_expit,
        scs.expit,
        {"x": np.linspace(-200, 200, 1000)}
    ),
    "gaussian": (
        gaussian,
        stats.norm.pdf,
        {"data": np.linspace(-5, 5, 100), "mean": 0.0, "sig": 1.0}
    ),
    "trunc_gaussian": (
        trunc_gaussian,
        comp_truncnorm,
        {"data": np.linspace(2, 8, 100), "mean": 5.0, "sig": 2.0, "lower": 1.0, "upper": 9.0}
    ),
    "powerlaw": (
        powerlaw,
        comp_powerlaw,
        {"data": np.linspace(2, 150, 100), "slope": -2.35, "minimum": 3.0, "maximum": 100.0}
    ),
    "powerlaw_close_to_1": (
        powerlaw,
        comp_powerlaw,
        {"data": np.linspace(2, 150, 100), "slope": 1e-6 - 1, "minimum": 3.0, "maximum": 100.0}
    ),
    "beta_spin": (
        beta_spin,
        stats.beta.pdf,
        {"spin_mag": np.linspace(0.01, 0.99, 100), "alpha": 2.0, "beta": 3.0}
    ),
    "powerlaw+peak": (
        PowerlawPlusPeak_PrimaryMass, 
        comp_powerlawpeak,
        {'data': {'mass_1': np.linspace(3, 100, 100)}, "alpha": 3, "minimum": 3, "maximum": 85, "delta_m": 5, "mpp": 35, "sigpp": 5, "lam": 0.05}, 
    ),
    "brokenpowerlaw+2peaks": (
        BrokenPowerlawPlusTwoPeaks_PrimaryMass, 
        comp_bpl2p,
        {'data': {'mass_1': np.linspace(3, 300, 2000)}, "alpha_1": 1.1, "alpha_2": 3, "mmin": 4.9, "mmax": 300, "break_mass": 35, "delta_m_1": 3, 
         "lam_fractions": (0.5,0.4,0.1), "mpp_1": 10, "sigpp_1": 1.5, "mpp_2": 35, "sigpp_2": 5}, 
    ),
    "redshift psi": (
        PowerlawRedshiftPsi,
        comp_powerlawredshiftpsi,
        {'data': {'redshift': np.linspace(0.001, 1.5, 1000)}, 'lamb': 2.0}
    ),
    "spins beta": (
        iid_beta_spin,
        comp_spins_beta,
        {'data': {'a_1': np.linspace(1e-5, 1-1e-5, 100), 'a_2': 1 - np.linspace(1e-5, 1-1e-5, 100)}, 'mu': 0.2, 'var': 0.1} 
    ),
    "spins gauss": (
        iid_normal_spin,
        comp_spins,
        {'data': {'a_1': np.linspace(1e-5, 1-1e-5, 100), 'a_2': 1 - np.linspace(1e-5, 1-1e-5, 100)}, 'mu': 0.2, 'var': 0.1} 
    ),
    "tilts iso+gauss": (
        tilt_model,
        comp_tilts,
        {'data': {'cos_tilt_1': np.linspace(-1,1,100), 'cos_tilt_2': -np.linspace(-1,1,100)}, 'mu': 0.5, 'sig': 0.2, 'zeta': 0.4}
    ),
    "chi_eff gauss": (
        chieff_gaussian,
        comp_gaussian_chieff,
        {'data': {'chi_eff': np.linspace(-1+1e-5, 1-1e-5, 1000)}, 'mean': 0.05, 'sig': 0.2}
    ),
    "chi_p gauss": (
        chip_gaussian,
        comp_gaussian_chip,
        {'data': {'chi_p': np.linspace(1e-5, 1-1e-5, 1000)}, 'mean': 0.25, 'sig': 0.2}
    ),
}

@pytest.mark.parametrize("model_name", TEST_MODELS.keys())
def test_against_standard_libraries(model_name):
    jax_func, ref_func, kwargs = TEST_MODELS[model_name]

    jax_result = np.exp(jax_func(**kwargs))
    ref_result = ref_func(*kwargs.values())

    np.testing.assert_allclose(
        jax_result,
        ref_result,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Mismatch found in model: {model_name}"
    )


@pytest.mark.parametrize("model_name", TEST_MODELS.keys())
def test_jittable_and_differentiable(model_name):
    """Each model must be jit-compilable, jax.grad-able, and produce no NaN/inf
    in either its forward output or in the gradient w.r.t. its float hyperparameters."""
    jax_func, _ref, kwargs = TEST_MODELS[model_name]
    keys = list(kwargs.keys())

    # Identify the float-valued (or tuple-of-float) hyperparameters to grad over.
    float_keys, float_args = [], []
    for k in keys[1:]:
        v = kwargs[k]
        if isinstance(v, tuple) and all(isinstance(x, (int, float)) for x in v):
            float_keys.append(k)
            float_args.append(jnp.asarray(v, dtype=jnp.float32))
        elif isinstance(v, (int, float, np.floating, np.integer)):
            float_keys.append(k)
            float_args.append(jnp.asarray(v, dtype=jnp.float32))

    def loss(*float_args):
        kw = dict(kwargs)
        for k, v in zip(float_keys, float_args):
            kw[k] = tuple(v) if isinstance(kwargs[k], tuple) else v
        out = jax_func(**kw)
        # Mask non-finite (out-of-support -inf) entries so the sum stays finite
        # and -inf doesn't propagate into NaN gradients.
        return jnp.sum(jnp.where(jnp.isfinite(out), out, 0.0))

    # Forward (eager) must produce no NaN.
    out = jax_func(**kwargs)
    assert not bool(jnp.any(jnp.isnan(out))), f"{model_name}: forward has NaN"

    # If there are no float hyperparams, jit-compile the bare forward and stop.
    if not float_keys:
        _ = jax.jit(jax_func)(**kwargs)
        return

    # jit(grad(...)) — single compile that exercises both jit and grad.
    grad_fn = jax.jit(jax.grad(loss, argnums=tuple(range(len(float_keys)))))
    grads = grad_fn(*float_args)
    for k, g in zip(float_keys, grads):
        g_arr = np.asarray(g)
        assert not np.any(np.isnan(g_arr)), f"{model_name}: NaN in d/d{k} = {g_arr}"
        assert not np.any(np.isinf(g_arr)), f"{model_name}: inf in d/d{k} = {g_arr}"
