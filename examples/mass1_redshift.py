'''
Example script for running PixelPop to infer correlated population in primary mass and redshift,
based on GWTC-3 data. 

Author: Jack Heinzel
'''

import jax
import numpyro
import numpy as np
from jax import numpy as jnp
import pandas as pd
import pixelpop
import h5py

varcut = 1
mmin = 3

# dictionary of injections, containing 1D arrays. Must contain 
# 'prior', 'total_generated' and 'analysis_time' keys in addition to GW parameters
# adapted from publicly available GW injections at https://zenodo.org/records/7890398
with h5py.File('data/GWTC3_injections.h5') as f:
    injections = {k: f[k][()] for k in f.keys()}
    
# dictionary of gw samples, containing 2D arrays of shape (Nobs, NPE). Must contain 'prior' key 
# in addition to GW parameters
# adapted from publicly available GW posteriors at https://zenodo.org/records/6513631 and https://zenodo.org/records/8177023
with h5py.File('data/GWTC3_posteriors.h5') as f:
    _posteriors = {k: f[k][()] for k in f.keys()}
    
keys = ['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift', 'prior']
posteriors = {k: jnp.array([p[k] for p in _posteriors]) for k in keys}

print(f"I have {posteriors['mass_1'].shape[0]} events")

def clean_data(data, min_m=mmin, max_m=200, max_z=2.3, remove=False):
    pixelpop.utils.data.clean_par(data, 'log_mass_1', jnp.log(min_m), jnp.log(max_m), remove=remove)
    pixelpop.utils.data.clean_par(data, 'redshift', 0., max_z, remove=remove)
    
# convert to log m1, log m2 space

posteriors = pixelpop.utils.data.convert_m1_to_lm1(posteriors)
injections = pixelpop.utils.data.convert_m1_to_lm1(injections)

clean_data(posteriors)
clean_data(injections, remove=True)

parameters = ['log_mass_1', 'redshift'] # parameters of "PixelPop"-ed space
other_parameters = ['mass_ratio', 'a', 't'] # nuisance parameters, we define parametric population model

# Pin these parametric hyperparameters to set values
priors = {'max_z': [[2.3], numpyro.distributions.Delta], 'qmin': [[0.1], numpyro.distributions.Delta]}

pp_data = pixelpop.utils.PixelPopData(
    posteriors = posteriors, # individual GW parameters
    injections = injections, # injections to estimate selection effects
    pixelpop_parameters = parameters, # parameters to infer with PixelPop ICAR model
    other_parameters = other_parameters, # "nuisance" parameters
    bins = 100, # number of bins along each axis
    minima = {'log_mass_1': np.log(mmin)}, # minimum of space
    maxima = {}, # maxima of space, set to defaults
    parametric_models = {}, # use defaults
    priors = priors, # modify prior on max_z and qmin
    lower_triangular = False, # full space is allowed
    marginalize_sigma = True, # use analytic marginalization of sigma coupling strength
)

probabilistic_model, initial_value = pixelpop.models.probabilistic.setup_probabilistic_model(
    pp_data
    )

# run the inference, hyperparameters of NUTS (warmup, maxtreedepth, etc.) tuned somewhat
output = pixelpop.models.probabilistic.inference_loop(
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
    run_dir='../../results/',
    name=f'm1z_varcut{varcut}',
    print_keys=['Nexp', 'log_likelihood', 'log_likelihood_variance', 'lnsigma', 'beta'], 
    # ^ for checking on chains while they run
    dense_mass=False
    )

# save output in popsummary file
pixelpop.result.create_popsummary(
    pp_data, 
    output, # results
    f'm1z_varcut{varcut}', # name of file
    popsummary_path='results/popsummary/',
    datadir='data',
    metadata_label="bbh", # doesn't exist, will skip metadata saving
    overwrite=True,
    )