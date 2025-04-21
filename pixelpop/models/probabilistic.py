import numpy as np
from ..utils.nearest_neighbor import create_CAR_coupling_matrix
from .gwpop_models import * 
from .car import initialize_ICAR
import numpyro.distributions as dist
import jax.numpy as jnp
from ..utils.data import place_in_bins
from jax.scipy.special import logsumexp as LSE
from numpyro.infer import MCMC, NUTS
from tqdm import tqdm
import sys
from numpyro.diagnostics import summary, print_summary
import pickle as pkl
from jax import random

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

hyperparameters_plausible = {
    'alpha':3, 'beta':2, 'mmin':2, 'mmax':199, 'delta_m':5, 'mpp':35, 'sigpp':5, 
    'lam':0.005, 'lamb':2, 'mu_x':0.06, 'sig_x':0.1, 'mu_spin':0.2, 'var_spin':0.1, 
    'mu_tilt':0.6, 'sig_tilt':0.6, 'zeta_tilt':0.5, 'lnsigma':-1, 'lncor': -5, 
    'mean': 0, 'qmin': 0.02, 'max_z': 2.4,
}

parameter_values = {'mass_1': 40., 'log_mass_1': np.log(40.), 'mass_ratio': 0.9, 'chi_eff': 0., 'redshift': 0.2, 'a_1': 0.2, 'a_2': 0.2, 'cos_tilt_1': 0., 'cos_tilt_2': 0.}

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
    'alpha': ([-4, 12], dist.Uniform), 'beta': ([-2, 7], dist.Uniform), 'qmin': ([0, 1], dist.Uniform), 'mmin': ([2, 10], dist.Uniform), 'mmax': ([60, 200], dist.Uniform), 
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

    def get_initial_value(plausible_hyperparameters, parameters, Nobs, inj_weights):
        bin_med = [(bin_axes[ii][:-1] + bin_axes[ii][1:])/2 for ii in range(dimension)]
        # print(bin_med)
        interpolation_grid = np.meshgrid(*bin_med, indexing='ij')
        data_grid = {p.replace('_psi',''): interpolation_grid[ii] for ii, p in enumerate(parameters)}    
        
        initial_interpolation = np.sum([
            parameter_to_gwpop_model[p](data_grid, *[plausible_hyperparameters[h] for h in parameter_to_hyperparameters[p]]) for ii, p in enumerate(parameters)
        ], axis=0)
        pdet = LSE(initial_interpolation[inj_bins] + inj_weights) - jnp.log(injections['total_generated'])
        Rexp = jnp.log(Nobs) - pdet - jnp.log(injections['analysis_time'])
        initial_interpolation = np.logaddexp(initial_interpolation, -10*np.ones_like(initial_interpolation)) # logaddexp -10 to smooth out negative divergences
        return {'merger_rate_density': Rexp + initial_interpolation}

    parameters_psi = [p.replace('redshift', 'redshift_psi') for p in parameters]
    initial_value = get_initial_value(hyperparameters_plausible, parameters_psi, event_bins[0].shape[0], inj_ln_dVTc-injections['log_prior'])

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

    def probabilistic_model(posteriors, injections):
        
        event_weights, inj_weights = nonparametric_model(event_bins, inj_bins, event_ln_dVTc-posteriors['log_prior'], inj_ln_dVTc-injections['log_prior'])
        event_weights, inj_weights = parametric_model(posteriors, injections, event_weights, inj_weights)

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

    return probabilistic_model, initial_value

def get_worst_rhat_neff(chain_samples):
    chain_summary = summary(chain_samples)
    rhats = [[key, chain_summary[key]['r_hat']] for key in chain_summary]
    neffs = [[key, chain_summary[key]['n_eff']] for key in chain_summary]
    
    name, pos, rhat_values = [], [], []
    for rh in rhats:
        ind = np.unravel_index(np.argmax(rh[1], axis=None), rh[1].shape)
        name.append(rh[0])
        pos.append(ind)
        rhat_values.append(rh[1][ind])
    
    worst_rhat = np.argmax(rhat_values)
    rhat_chain = chain_samples[name[worst_rhat]][...,*pos[worst_rhat]]
    rhat_key = f'{name[worst_rhat]}{list(pos[worst_rhat])}'.replace('[]','')

    name, pos, neff_values = [], [], []
    for rh in neffs:
        ind = np.unravel_index(np.argmin(rh[1], axis=None), rh[1].shape)
        name.append(rh[0])
        pos.append(ind)
        neff_values.append(rh[1][ind])
    
    worst_neff = np.argmin(neff_values)
    neff_chain = chain_samples[name[worst_neff]][...,*pos[worst_neff]]
    neff_key = f'{name[worst_neff]}{list(pos[worst_neff])}'.replace('[]','')
    return rhat_key, rhat_chain, neff_key, neff_chain


def inference_loop(
    probabilistic_model, model_kwargs={}, initial_value={}, warmup=10000, tot_samples=100, thinning=100, pacc=0.65, maxtreedepth=10, 
    num_samples=1, parallel=1, rng_key=random.PRNGKey(1), cache_cadence=1, name='',
    print_keys=['Nexp', 'log_likelihood', 'log_likelihood_variance']
    ):

    kernel = NUTS(probabilistic_model, max_tree_depth=maxtreedepth, target_accept_prob=pacc, init_strategy=numpyro.infer.init_to_value(values=initial_value))

    samples = None
    load_pkl = False

    for chain in range(parallel):
        print(f"Warming up chain #{chain + 1} out of {parallel}")
        mcmc = MCMC(kernel, thinning=thinning, num_warmup=warmup, num_samples=num_samples*thinning, num_chains=1)# , chain_method='vectorized')# , chain_method='sequential') # vectorized is an experimental method. We can pass 'parallel' which attempts to distribute the chains across multiple GPUs, e.g. on pcdev12 we could do num_chains = 4 across the a100s. If num_chains is too large, it defaults to 'sequential' which simply evaluates the chains in series.
        rng_key_, rng_key = random.split(rng_key)
        mcmc.run(rng_key, **model_kwargs)#, extra_fields=('~z.cell_locations_unordered', '~z.cell_locations_all'))
        first_sample = mcmc.get_samples()
        chain_samples = {key:np.array(first_sample[key])[None,...] for key in first_sample}
        table_size = len(print_keys) + 2
        # mcmc.thinning = 10
        # mcmc.num_samples = 100
        sys.stdout.write("\n"*(table_size+3)) # buffer line between the progress bars

        mcmc.transfer_states_to_host()
        # mcmc.progress_bar = False
        sample_iterator = tqdm(range(int(tot_samples/num_samples)+1))
        sample_iterator.set_description("drawing thinned samples")
        for sample in sample_iterator:
            mcmc.post_warmup_state = mcmc.last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, **model_kwargs)
            next_sample = mcmc.get_samples()
            sys.stdout.write("\x1b[1A\n\x1b[1A")
            #print([[key,chain_samples[key].shape] for key in chain_samples])
            for key in chain_samples:
                chain_samples[key] = np.concatenate((chain_samples[key], np.array(next_sample[key])[None,...]), axis=1)
            mcmc.transfer_states_to_host()
            if (sample % cache_cadence == 0) and (chain_samples[key].shape[1] >= 4):
                # print('\n')
                sys.stdout.write(f"\x1b[1A\x1b[2K"*(table_size+3)) # move the cursor up to overwrite the summary table for the NEXT print
                #sys.stdout.write(f"\n"*(table_size+3)) # clear the table?
                #sys.stdout.write(f"\x1b[{table_size+3}A") # move the cursor up to overwrite the summary table for the NEXT print
                
                rhat, rhat_chain, neff, neff_chain = get_worst_rhat_neff(chain_samples)
                summary_dict = {key: chain_samples[key] for key in print_keys}
                summary_dict['worst r_hat: '+rhat] = rhat_chain
                summary_dict['worst n_eff: '+neff] = neff_chain
                
                print_summary(summary_dict)
                
                with open(f'chain_{int(rng_key[1])}_{name}_samples.pkl', 'wb') as ff:
                    pkl.dump(chain_samples, ff)

                with open(f'chain_{int(rng_key[1])}_{name}_mcmc.pkl', 'wb') as ff:
                    pkl.dump(mcmc, ff)
                    
        if samples is None:
            samples = chain_samples.copy()
        else:
            for key in samples:
                samples[key] = np.concatenate((samples[key], chain_samples[key]), axis=0)

        return samples, mcmc