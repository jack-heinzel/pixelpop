import pytest
import numpy as np
import jax.numpy as jnp
import scipy.special as scs
import scipy.stats as stats

# If testing against gwpopulation for powerlaw/smoothing:
from gwpopulation.utils import powerlaw as gwpop_powerlaw
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift


from o4_default_mass_model import TwoPeakBrokenPowerLawSmoothedMassDistribution
# Assuming your models are saved in a file named `gwpop_models.py`
from pixelpop.models.gwpop_models import (
    log_expit, gaussian, trunc_gaussian, powerlaw, beta_spin, PowerlawPlusPeak_PrimaryMass, 
    BrokenPowerlawPlusTwoPeaks_PrimaryMass, PowerlawRedshift, PowerlawRedshiftPsi, 
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

def comp_powerlawredshift(data, lamb):
    zmodel = PowerLawRedshift(z_max=1.9, cosmo_model="FlatLambdaCDM")
    return zmodel(data, lamb=lamb, H0=67.90, Om0=0.3065)

def comp_powerlawredshiftpsi(data, lamb):
    zmodel = PowerLawRedshift(z_max=1.9)# , cosmo_model="FlatLambdaCDM")
    return zmodel.psi_of_z(data['redshift'], lamb=lamb)

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
        {"data": np.linspace(5, 50, 100), "slope": -2.35, "minimum": 2.0, "maximum": 100.0}
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
        {'data': {'mass_1': np.linspace(3, 300, 2000)}, "alpha_1": 1, "alpha_2": 3, "mmin": 4.9, "mmax": 300, "break_mass": 35, "delta_m_1": 3, 
         "lam_fractions": (0.5,0.4,0.1), "mpp_1": 10, "sigpp_1": 1.5, "mpp_2": 35, "sigpp_2": 5}, 
    ),
    "redshift psi": (
        PowerlawRedshiftPsi,
        comp_powerlawredshiftpsi,
        {'data': {'redshift': np.linspace(0.001, 1.5, 1000)}, 'lamb': 2.0}
    ),
    # more here, spins spin tilts
}

@pytest.mark.parametrize("model_name", TEST_MODELS.keys())
def test_against_standard_libraries(model_name):
    jax_func, ref_func, kwargs = TEST_MODELS[model_name]
   
    jax_result = np.exp(jax_func(**kwargs))
    ref_result = ref_func(*kwargs.values())
    
    # valid_mask = jnp.isfinite(jax_result)
    print(kwargs.get('data'))
    print(jax_result / ref_result)
    np.testing.assert_allclose(
        jax_result, # [valid_mask], 
        ref_result, # [valid_mask], 
        rtol=1e-5, 
        atol=1e-6,
        err_msg=f"Mismatch found in model: {model_name}"
    )
