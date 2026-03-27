import inspect
from gwpopulation.utils import xp
from gwpopulation.models.mass import double_power_law_primary_mass, truncnorm, BaseSmoothedMassDistribution

def four_component_double_power_law_primary_mass(
    mass, alpha_1, alpha_2, mmin, mmax, break_mass, lam_0, lam_1, lam_2, 
        mpp_1, sigpp_1, mpp_2, sigpp_2, mpp_3, sigpp_3, gaussian_mass_maximum=100):
    """
    A four-component double power law mass model: broken power law, three Gaussians.
    
    Parameters
    ----------
    mass: array-like
        The masses at which to evaluate the model (:math:`m`).
    alpha_1: float
        The power-law index below break (:math:`\alpha_1`).
    alpha_2: float
        The power-law index above break (:math:`\alpha_2`).
    mmin: float
        The minimum mass (:math:`m_{\min}`).
    mmax: float
        The maximum mass (:math:`m_{\max}`).
    break_mass: float
        The mass at which the break occurs (:math:`\delta`).
    lam_0: float
        The fraction of black holes in the power law (:math:`\hat{\lambda}_0`).
    lam_1: float
        The fraction of black holes in the lower Gaussian component (:math:`\hat{\lambda}_1`).
    lam_2: float
        The fraction of black holes in the next highest Gaussian component (:math:`\hat{\lambda}_2`).
    mpp_1: float
        Mean of the lowest mass Gaussian component (:math:`\mu_{m, 1}`).
    mpp_2: float
        Mean of the next highest mass Gaussian component (:math:`\mu_{m, 2}`).
    mpp_3: float
        Mean of the highest mass Gaussian component (:math:`\mu_{m, 3}`).
    sigpp_1: float
        Standard deviation of the lowest mass Gaussian component (:math:`\sigma_{m, 1}`).
    sigpp_2: float
        Standard deviation of the next highest mass Gaussian component (:math:`\sigma_{m, 2}`).
    sigpp_3: float
        Standard deviation of the highest mass Gaussian component (:math:`\sigma_{m, 3}`).
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
        Note that this applies the same value to both.
    """
    lam_3 = 1 - lam_2 - lam_1 - lam_0
    break_fraction = (break_mass  - mmin) / (mmax - mmin)
    p_pow = double_power_law_primary_mass(mass, alpha_1=alpha_1, alpha_2=alpha_2, mmin=mmin, mmax=mmax, break_fraction=break_fraction)
    p_norm1 = truncnorm(
        mass, mu=mpp_1, sigma=sigpp_1, high=gaussian_mass_maximum, low=mmin
    )
    p_norm2 = truncnorm(
        mass, mu=mpp_2, sigma=sigpp_2, high=gaussian_mass_maximum, low=mmin
    )
    p_norm3 = truncnorm(
        mass, mu=mpp_3, sigma=sigpp_3, high=gaussian_mass_maximum, low=mmin
    )

    prob = lam_0 * p_pow +  lam_1 * p_norm1 + lam_2 * p_norm2 + lam_3 * p_norm3
    return prob

def three_component_double_power_law_primary_mass(
    mass, alpha_1, alpha_2, mmin, mmax, break_mass, lam_0, lam_1, mpp_1, sigpp_1, mpp_2, sigpp_2, gaussian_mass_maximum=100
    ):
    """
    A three-component double power law mass model: broken power law, two Gaussians.
    
    Parameters
    ----------
    mass: array-like
        The masses at which to evaluate the model (:math:`m`).
    alpha_1: float
        The power-law index below break (:math:`\alpha_1`).
    alpha_2: float
        The power-law index above break (:math:`\alpha_2`).
    mmin: float
        The minimum mass (:math:`m_{\min}`).
    mmax: float
        The maximum mass (:math:`m_{\max}`).
    break_mass: float
        The mass at which the break occurs (:math:`\delta`).
    lam_0: float
        The fraction of black holes in the power law (:math:`\hat{\lambda}_0`).
    lam_1: float
        The fraction of black holes in the lower Gaussian component (:math:`\hat{\lambda}_1`).
    mpp_1: float
        Mean of the lower mass Gaussian component (:math:`\mu_{m, 1}`).
    mpp_2: float
        Mean of the higher mass Gaussian component (:math:`\mu_{m, 2}`).
    sigpp_1: float
        Standard deviation of the lower mass Gaussian component (:math:`\sigma_{m, 1}`).
    sigpp_2: float
        Standard deviation of the higher mass Gaussian component (:math:`\sigma_{m, 2}`).
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
        Note that this applies the same value to both.
    """
    lam_2 = 1 - lam_1 - lam_0
    return four_component_double_power_law_primary_mass(
        mass, alpha_1=alpha_1, alpha_2=alpha_2, mmin=mmin, mmax=mmax, break_mass=break_mass, 
        lam_0=lam_0, lam_1=lam_1, lam_2=lam_2, mpp_1=mpp_1, sigpp_1=sigpp_1, mpp_2=mpp_2, sigpp_2=sigpp_2, 
        mpp_3=0, sigpp_3=1, gaussian_mass_maximum=gaussian_mass_maximum
    )

def two_component_double_power_law_primary_mass(
    mass, alpha_1, alpha_2, mmin, mmax, break_mass, lam_0, mpp_1, sigpp_1, gaussian_mass_maximum=100
    ):
    """
    A two-component double power law mass model: broken power law, one Gaussian.

    Parameters
    ----------
    mass: array-like
    
    alpha_1: float
        The power-law index below break (:math:`\alpha_1`).
    alpha_2: float
        The power-law index above break (:math:`\alpha_2`).
    mmin: float
        The minimum mass (:math:`m_{\min}`).
    mmax: float
        The maximum mass (:math:`m_{\max}`).
    break_mass: float
        The mass at which the break occurs (:math:`\delta`).
    lam_0: float
        The fraction of black holes in the power law (:math:`\hat{\lambda}_0`).
    mpp_1: float
        Mean of the Gaussian component (:math:`\mu_{m, 1}`).
    sigpp_1: float
        Standard deviation of the Gaussian component (:math:`\sigma_{m, 1}`).
    gaussian_mass_maximum: float, optional
    """
    lam_1 = 1 - lam_0

    return four_component_double_power_law_primary_mass(
        mass, alpha_1=alpha_1, alpha_2=alpha_2, mmin=mmin, mmax=mmax, break_mass=break_mass, lam_0=lam_0, lam_1=lam_1, lam_2=0,
        mpp_1=mpp_1, sigpp_1=sigpp_1, mpp_2=0, sigpp_2=1, mpp_3=0, sigpp_3=1, gaussian_mass_maximum=gaussian_mass_maximum
    )

class BrokenPowerLawPlusPeaksSmoothedMassDistribution(BaseSmoothedMassDistribution):
    """
    Broken power law mass distribution with Gaussian components with smoothing.


    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha_1: float
        Power law exponent of the primary mass distribution below the break.
    alpha_2: float
        Power law exponent of the primary mass distribution above the break.
    beta: float
        Power law exponent of the mass ratio distribution.
    mmin_1: float
        Minimum primary black hole mass.
    mmin_2: float
        Minimum secondary black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    break_mass: float
        Mass at which the power law transitions from alpha_1 to alpha_2.
    lam_0: float
        Fraction of black holes in the power law component.
    lam_1: float
        Fraction of black holes in the lower mass Gaussian component.
    mpp_1: float
        Mean of the lower mass Gaussian component.
    mpp_2: float
        Mean of the higher mass Gaussian component.
    sigpp_1: float
        Standard deviation of the lower mass Gaussian component.
    sigpp_2: float
        Standard deviation of the higher mass Gaussian component.
    delta_m_1: float
        Rise length of the low end of the primary mass distribution.
    delta_m_2: float
        Rise length of the secondary mass distribution.

    Notes
    -----
    The Gaussian components are bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = None #Replace in subclass

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)
    
    def __init__(self, mmin=2, mmax=200, normalization_shape=(1000, 500), cache=True, spacing="log"):
        self.mmin = mmin
        self.mmax = mmax
        if spacing == "log":
            self.m1s = xp.logspace(xp.log10(mmin), xp.log10(mmax), normalization_shape[0])
        elif spacing == "linear":
            self.m1s = xp.linspace(mmin, mmax, normalization_shape[0])
        self.spacing = spacing
        self.qs = xp.linspace(0.001, 1, normalization_shape[1])
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.cache = cache

    def __call__(self, dataset, *args, **kwargs):
        beta = kwargs.pop("beta")
        mmin_1 = kwargs.pop("mlow_1", self.mmin)
        mmin_2 = kwargs.pop("mlow_2", self.mmin)
        delta_m_1 = kwargs.pop("delta_m_1", 0)
        delta_m_2 = kwargs.pop("delta_m_2", 0)
        mmax = kwargs.get("mmax", self.mmax)
        if "jax" not in xp.__name__:
            if mmin_1 < self.mmin or mmin_2 < self.mmin:
                raise ValueError(
                    "{self.__class__}: mlow ({mmin}) < self.mmin ({self.mmin})"
                )
            if mmax > self.mmax:
                raise ValueError(
                    "{self.__class__}: mmax ({mmax}) > self.mmax ({self.mmax})"
                )
        p_m1 = self.p_m1(dataset, mmin=mmin_1, delta_m=delta_m_1, **kwargs, **self.kwargs)
        p_q = self.p_q(dataset, beta=beta, mmin=mmin_2, delta_m=delta_m_2)
        prob = p_m1 * p_q
        return prob
    
    @property
    def variable_names(self):
        vars = getattr(
            self.primary_model,
            "variable_names",
            inspect.getfullargspec(self.primary_model).args[1:],
        )
        vars += ["beta", "delta_m_1", "delta_m_2", "mlow_1", "mlow_2"]
        vars.remove("mmin")
        vars = set(vars).difference(self.kwargs.keys())
        return vars
    
    def _cache_q_norms(self, masses):
        """
        Cache the information necessary for linear interpolation of the mass
        ratio normalisation
        """
        from gwpopulation.models.interped import _setup_interpolant
        if self.spacing == 'log':
            self._q_interpolant = _setup_interpolant(
                xp.log(self.m1s), xp.log(masses), kind="linear", backend=xp
            )
        else:
            self._q_interpolant = _setup_interpolant(
                self.m1s, masses, kind="linear", backend=xp
            )
    
class OnePeakBrokenPowerLawSmoothedMassDistribution(BrokenPowerLawPlusPeaksSmoothedMassDistribution):
    primary_model = two_component_double_power_law_primary_mass

class TwoPeakBrokenPowerLawSmoothedMassDistribution(BrokenPowerLawPlusPeaksSmoothedMassDistribution):
    primary_model = three_component_double_power_law_primary_mass

class ThreePeakBrokenPowerLawSmoothedMassDistribution(BrokenPowerLawPlusPeaksSmoothedMassDistribution):
    primary_model = four_component_double_power_law_primary_mass
    

def mlow_2_condition(reference_params, mlow_1):
    return dict(
        minimum=reference_params["minimum"],
        maximum=mlow_1
        )
