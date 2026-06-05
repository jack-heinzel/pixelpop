'''
Example script for running PixelPop to infer correlated population in primary and secondary masses,
based on GWTC-3 data. Highlights the "lower-triangular" keyword option for restricting the domain.

Author: Jack Heinzel
'''

import jax

import pixelpop
import numpy as np
from jax import numpy as jnp
import pickle as pkl
from numpyro import distributions as dist
import h5py

varcut = 1
mmin = 3

# define bounds of space if different from defaults
minima = {'mass_1': mmin}
maxima = {'mass_1': 200.}

# dictionary of injections, containing 1D arrays. Must contain 
# 'prior', 'total_generated' and 'analysis_time' keys in addition to GW parameters

# adapted from publicly available GW injections at https://zenodo.org/records/7890398
with h5py.File('data/GWTC3_injections.h5') as f:
    injections = {k: f[k][()] for k in f.keys()}
    
# Different events have different numbers of PE samples, so we stack them into a single
# rectangular (Nobs, NPE) dict with pixelpop.utils.data.posteriors_to_rectangular: events
# with fewer than NPE samples are padded with prior=+inf rows (zero weight), and the real
# per-event sample counts are returned as `event_counts` (passed to PixelPopData so the
# variance / Neff calculations use the right N). Here we pad up to the largest event so no
# samples are discarded.
# adapted from publicly available GW posteriors at https://zenodo.org/records/6513631 and https://zenodo.org/records/8177023
with h5py.File('data/GWTC3_posteriors.h5') as f:
    _posteriors = {event: f[event][()] for event in f.keys()}

keys = ['mass_1', 'mass_2', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift']
npe = max(_posteriors[event]['prior'].shape[0] for event in _posteriors)
posteriors, event_counts, event_names = pixelpop.utils.data.posteriors_to_rectangular(
    _posteriors, keys, npe
)


with open('../../data/all_cbc/data/injections.pkl', 'rb') as ff:
    injections = pkl.load(ff)

print(f"I have {len(event_names)} events, padded to {npe} samples each")

def clean_data(data, min_m=mmin, max_m=200, max_z=2.3, remove=False):
    pixelpop.utils.data.clean_par(data, 'log_mass_1', jnp.log(min_m), jnp.log(max_m), remove=remove)
    pixelpop.utils.data.clean_par(data, 'log_mass_2', jnp.log(min_m), jnp.log(max_m), remove=remove)
    pixelpop.utils.data.clean_par(data, 'redshift', 0., max_z, remove=remove)

# convert to log m1, log m2 space

posteriors = pixelpop.utils.data.convert_m1m2_to_lm1lm2(posteriors)
injections = pixelpop.utils.data.convert_m1m2_to_lm1lm2(injections)

clean_data(posteriors)
clean_data(injections, remove=True)

parameters = ['log_mass_1', 'log_mass_2']
other_parameters = ['redshift', 'a', 't']

priors = {'max_z': [[2.3], dist.Delta]}

pp_data = pixelpop.utils.PixelPopData(
    posteriors = posteriors, # individual GW parameters
    injections = injections, # injections to estimate selection effects
    pixelpop_parameters = parameters, # parameters to infer with PixelPop ICAR model
    other_parameters = other_parameters, # "nuisance" parameters
    bins = 100, # number of bins along each axis
    minima = {'log_mass_1': jnp.log(mmin), 'log_mass_2': jnp.log(mmin)}, # minimum of space
    maxima = {'log_mass_1': jnp.log(200), 'log_mass_2': jnp.log(200)}, # maxima of space
    parametric_models = {}, # use defaults
    priors = priors, # modify prior on max_z
    lower_triangular = True, # Restrict domain to m1 > m2
    marginalize_sigma = True, # use analytic marginalization of sigma coupling strength
    event_counts = event_counts, # real per-event sample counts (padded rows excluded)
)

probabilistic_model, initial_value = pixelpop.models.probabilistic.setup_probabilistic_model(
    pp_data
    )

# run the inference, hyperparameters of NUTS (warmup, maxtreedepth, etc.) tuned somewhat    
output, mcmc = pixelpop.models.probabilistic.inference_loop(
    probabilistic_model, 
    model_kwargs={'posteriors': pp_data.posteriors, 'injections': pp_data.injections}, 
    initial_value=initial_value,
    warmup=500_000, 
    tot_samples=1000,
    thinning=1000,
    pacc=0.65,
    maxtreedepth=5,
    num_samples=4,
    parallel=5,
    run_dir='results/',
    name=f'm1m2_varcut{varcut}',
    print_keys=['Nexp', 'log_likelihood', 'log_likelihood_variance', 'lamb', 'mu_spin', 'var_spin'],
    # ^ for checking on chains while they run
    dense_mass=False
    )

# save output in popsummary file, also computes diagnostics for convergence 
# and Monte Carlo systematics
pixelpop.result.create_popsummary(
    pp_data, # store pixelpop relevant data
    output, # results 
    f'm1m2_varcut{varcut}', # name of file
    popsummary_path='results/popsummary/', 
    datadir='data',
    metadata_label="bbh", # doesn't exist, will skip metadata saving
    overwrite=True, 
    )

