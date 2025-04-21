import numpy as np
from ..utils.nearest_neighbor import create_CAR_coupling_matrix
from .gwpop_models import * 
from .car import initialize_ICAR
import numpyro.distributions as dist
import jax.numpy as jnp
from ..utils.data import place_in_bins
from jax.scipy.special import logsumexp as LSE

parameter_to_gwpop_model = {
    'mass_1': PowerlawPlusPeak_PrimaryMass, #(data, slope, minimum, maximum, delta_m, mpp, sigpp, lam)
    'log_mass_1': PowerlawPlusPeak_PrimaryMass, #(data, slope, minimum, maximum, delta_m, mpp, sigpp, lam)
    'mass_ratio': SimplePowerlaw_MassRatio, #(data, slope)
    'redshift': PowerlawRedshift, #(data, lamb, minimum, maximum, normalize=False):
    'redshift_psi': PowerlawRedshiftPsi, #(data, lamb, minimum, maximum, normalize=False):
    'chi_eff': chieff_gaussian, #(data, mean, sig)
    'spin': spin_default, #(data, mu, var, sig, zeta)
    'a': iid_beta_spin, #(data, mu, var)
    't': tilt_iid, #(data, mu, sig, zeta)
}

parameter_to_hyperparameters = {
    'mass_1': ['alpha', 'mmin', 'mmax', 'delta_m', 'mpp', 'sigpp', 'lam'], # slope, minimum, maximum, delta_m, mpp, sigpp, lam)
    'log_mass_1': ['alpha', 'mmin', 'mmax', 'delta_m', 'mpp', 'sigpp', 'lam'], # slope, minimum, maximum, delta_m, mpp, sigpp, lam)
    'mass_ratio': ['beta', 'qmin'], #(data, slope, minimum, delta_m)
    'redshift': ['lamb', 'max_z'],# 'z_minimum', 'z_maximum'], #(data, lamb, minimum, maximum, normalize=False):
    'redshift_psi': ['lamb', 'max_z'],# 'z_minimum', 'z_maximum'], #(data, lamb, minimum, maximum, normalize=False):
    'chi_eff': ['mu_x', 'sig_x'], #(data, mean, sig)
    'spin': ['mu_spin', 'var_spin', 'sig_tilt', 'zeta_tilt'], #(data, alpha, beta, sig, zeta)
    'a': ['mu_spin', 'var_spin'],
    't': ['mu_tilt', 'sig_tilt', 'zeta_tilt'],
}

default_priors = {
    'alpha': ([-4, 12], dist.Uniform), 'beta': ([-2, 7], dist.Uniform), 'qmin': ([0, 1], dist.Uniform), 'mmin': ([2, mmin_max], dist.Uniform), 'mmax': ([mmax_min, 100], dist.Uniform), 
    'delta_m': ([0, 10], dist.Uniform), 'mpp': ([20, 50], dist.Uniform), 'sigpp': ([1, 10], dist.Uniform), 'lam': ([0, 1], dist.Uniform), 
    'lamb': ([-2, 10], dist.Uniform), 'mu_x': ([-1, 1], dist.Uniform), 'sig_x': ([0.005, 1.], dist.Uniform), 'mu_spin': ([0, 1], dist.Uniform),
    'var_spin': ([0.005, 0.25], dist.Uniform), 'mu_tilt': ([-1, 1], dist.Uniform), 'sig_tilt': ([0.1, 4], dist.Uniform), 
    'zeta_tilt': ([0, 1], dist.Uniform), 'z_minimum': ([0.], dist.Delta), 'max_z': ([2.4], dist.Delta),
}


def setup_probabilistic_model(posteriors, injections, parameters, other_parameters, bins, minima={}, maxima={}, priors={}, UncertaintyCut=1.):
    # define model
    dimension = len(parameters)
    if np.ndim(bins) == 0:
        bins = [bins] * dimension
    adj_matrices = [create_CAR_coupling_matrix(bins[ii], 1, isVisible=False) for ii in range(dimension)]

    if 'redshift' in parameters:
        from astropy.cosmology import Planck15
        from astropy import units
        max_z = np.maximum(np.max(injections['redshift']), np.max(posteriors['redshift']))
        zs = np.linspace(1e-6, max_z, 10000)
        dVs = Planck15.differential_comoving_volume(zs) * 4 * np.pi * units.sr
        ln_dVTc = np.log(dVs.to(units.Gpc**3).value) - np.log(1 + zs)
        event_z = posteriors['redshift']
        inj_z = injections['redshift']
        event_ln_dVTc = jnp.array(np.interp(event_z, zs, ln_dVTc))
        inj_ln_dVTc = jnp.array(np.interp(inj_z, zs, ln_dVTc))
    else:
        event_ln_dVTc = jnp.zeros_like(posteriors['log_prior'])
        inj_ln_dVTc = jnp.zeros_like(injections['log_prior'])
        
    event_bins, inj_bins, bin_axes, logdV = place_in_bins(parameters, posteriors, injections, bins=bins, minima=minima, maxima=maxima)
    # default priors
    hyperparameter_priors = {}
    for p in other_parameters:
        for h in parameter_to_hyperparameters[p]:
            if h in priors:    
                pprint = priors[h]
                print(f'Using custom prior {h} = {pprint[1].__name__}({str(pprint[0])[1:-1]}) in {p} model')
                hyperparameter_priors[h] = priors[h]
            else:
                pprint = default_priors[h]
                print(f'Using default prior {h} = {pprint[1].__name__}({str(pprint[0])[1:-1]}) in {p} model')
                hyperparameter_priors[h] = default_priors[h]

    def parametric_model(data, injections, event_weights, inj_weights):
        sample = {}
        for key in hyperparameter_priors:
            args, distribution = hyperparameter_priors[key]
            sample[key] = numpyro.sample(key, distribution(*args))

        for p in other_parameters:
            event_weights += parameter_to_gwpop_model[p](data, *[sample[h] for h in parameter_to_hyperparameters[p]])
            inj_weights += parameter_to_gwpop_model[p](injections, *[sample[h] for h in parameter_to_hyperparameters[p]])
        return event_weights, inj_weights

    ICAR_length_scales = initialize_ICAR(dimension)

    def nonparametric_model(event_bins, inj_bins, event_weights, inj_weights):

        lsigmas = numpyro.sample('lnsigmas', dist.Uniform(-3,3), sample_shape=(dimension,)) # log_prob + prior_sampling_factor
        merger_rate_density = numpyro.sample('merger_rate_density', ICAR_length_scales(log_sigmas=lsigmas, single_dimension_adj_matrices=adj_matrices, is_sparse=True))
                    
        normalization = numpyro.deterministic('log_rate', LSE(merger_rate_density)+jnp.sum(logdV))
        for ii, p in enumerate(parameters):
            sum_axes = tuple(np.arange(dimension)[np.r_[0:ii,ii+1:dimension]])
            numpyro.deterministic(f'log_marginal_{p}', LSE(merger_rate_density-normalization, axis=sum_axes) + jnp.sum(logdV[:ii]) + jnp.sum(logdV[ii+1:]))

        event_weights += merger_rate_density[event_bins] # (69,3194)
        inj_weights += merger_rate_density[inj_bins]
        return event_weights, inj_weights

    def probabilistic_model(data, injections):
        # draw 2d population grid
        
        event_weights, inj_weights = nonparametric_model(event_bins, inj_bins, event_ln_dVTc-data['log_prior'], inj_ln_dVTc-injections['log_prior'])
        event_weights, inj_weights = parametric_model(data, injections, event_weights, inj_weights)

        ln_likelihood, nexp, pe_var, vt_var, total_var = rate_likelihood(event_weights, inj_weights, injections['total_generated'], live_time=injections['analysis_time'])
        # print(ln_likelihood, ln_likelihood_variance) # double check the uncertainty propagation AND the likelihood for rate
        taper = smooth(total_var, UncertaintyCut**2, 0.1) # "smooth" cutoff above Talbot+Golomb 2022 recommendation to retain autodifferentiability
        # print(ln_likelihood + taper)

        numpyro.deterministic("log_likelihood", ln_likelihood)
        numpyro.deterministic("log_likelihood_variance", total_var) # save these values!
        numpyro.deterministic("pe_variance", pe_var) # save these values!
        numpyro.deterministic("vt_variance", vt_var) # save these values!
        numpyro.deterministic("Nexp", nexp) # save these values!
        numpyro.factor("log_likelihood_plus_taper", ln_likelihood + taper)

    return probabilistic_model