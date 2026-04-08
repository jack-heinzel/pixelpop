from gwpopulation.models.mass import truncnorm

def independent_spin_magnitude_gaussian(dataset, mu_chi_1, mu_chi_2, sigma_chi_1, sigma_chi_2, amax_1, amax_2):
    """
    A model for the independent spin magnitude distribution of black holes.
    
    Parameters
    ----------
    dataset: dict
        Dictionary of arrays for 'a_1' and 'a_2'.
    mu_chi_1: float
        Mean of the spin magnitude Gaussian component for the primary black holes.
    mu_chi_2: float
        Mean of the spin magnitude Gaussian component for the secondary black holes.
    sigma_chi_1: float
        Standard deviation of the spin magnitude Gaussian component for the primary black holes.
    sigma_chi_2: float
        Standard deviation of the spin magnitude Gaussian component for the secondary black holes.
    amax_1: float
        Maximum spin magnitude for the primary black holes.
    amax_2: float
        Maximum spin magnitude for the secondary black holes.
    """
    p_a_1 = truncnorm(dataset["a_1"], mu=mu_chi_1, sigma=sigma_chi_1, high=amax_1, low=0)
    p_a_2 = truncnorm(dataset["a_2"], mu=mu_chi_2, sigma=sigma_chi_2, high=amax_2, low=0)
    return p_a_1 * p_a_2

def iid_spin_magnitude_gaussian(dataset, mu_chi, sigma_chi, amax):
    """
    A model for the independent and identically distributed spin magnitude distribution of black holes.
    
    Parameters
    ----------
    dataset: dict
        Dictionary of arrays for 'a_1' and 'a_2'.
    mu_chi: float
        Mean of the spin magnitude Gaussian component.
    sigma_chi: float
        Standard deviation of the spin magnitude Gaussian component.
    amax: float
        Maximum spin magnitude.
    """
    prior = independent_spin_magnitude_gaussian(dataset, mu_chi, mu_chi, sigma_chi, sigma_chi, amax, amax)
    return prior