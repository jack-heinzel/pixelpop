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
import pickle as pkl
import pixelpop
import h5py

varcut = 1
mmin = 3

with open('data/posteriors_chieff.pkl', 'rb') as ff:
    _posteriors = pkl.load(ff)

with open('data/injection_set_chieff.pkl', 'rb') as ff:
    injections = pkl.load(ff)

def clean_data(data, min_m=mmin, max_m=200, max_z=2.3, remove=False):
    pixelpop.utils.data.clean_par(data, 'log_mass_1', jnp.log(min_m), jnp.log(max_m), remove=remove)
    pixelpop.utils.data.clean_par(data, 'redshift', 0., max_z, remove=remove)
    
keys = ['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift', 'prior']
#posteriors = {k: jnp.array([p[k] for p in _posteriors]) for k in keys}

print(f"I have {_posteriors['mass_1'].shape[0]} events")
    
# convert to log m1, log m2 space

posteriors = pixelpop.utils.data.convert_m1_to_lm1(_posteriors)
injections = pixelpop.utils.data.convert_m1_to_lm1(injections)

clean_data(posteriors)
clean_data(injections, remove=True)

parameters = ['log_mass_1', 'redshift'] # parameters of "PixelPop"-ed space
other_parameters = ['mass_ratio', 'a', 't'] # nuisance parameters, we define parametric population model

# Pin these parametric hyperparameters to set values
priors = {'max_z': [[2.3], numpyro.distributions.Delta], 'qmin': [[0.1], numpyro.distributions.Delta]}

probabilistic_model, initial_value = pixelpop.models.probabilistic.setup_probabilistic_model(
    posteriors, # individual GW parameters
    injections, # injections to estimate selection effects
    parameters, # parameters to infer with PixelPop ICAR model
    other_parameters, # nuisance parameters
    100, # number of bins along each axis
    minima={'log_mass_1': np.log(mmin)}, # minimum of space
    maxima={}, # maxima set to default values
    priors=priors, # priors which differ from defaults
    UncertaintyCut=np.sqrt(varcut), # convergence criteria for likelihood estimator
    parametric_models={}, # parametric models for nuisance parameters are set to defaults
    length_scales=False, # same ICAR Gaussian coupling strength in all directions
    random_initialization=True, # initialize ICAR model from random Gaussian draw
    lower_triangular=False, # full space is allowed
    )

# run the inference, hyperparameters of NUTS (warmup, maxtreedepth, etc.) tuned somewhat
output = pixelpop.models.probabilistic.inference_loop(
    probabilistic_model, 
    model_kwargs={'posteriors': posteriors, 'injections': injections}, 
    initial_value=initial_value,
    warmup=500, 
    tot_samples=100,
    thinning=1,
    pacc=0.65,
    maxtreedepth=5,
    num_samples=1,
    parallel=1,
    run_dir='../../results/',
    name=f'm1z_varcut{varcut}',
    print_keys=['Nexp', 'log_likelihood', 'log_likelihood_variance', 'lnsigma', 'beta'], 
    # ^ for checking on chains while they run
    dense_mass=False
    )

# save output in popsummary file
pixelpop.result.create_popsummary(
    output, # results
    f'm1z_varcut{varcut}', # name of file
    parameters, # "PixelPop"-ed parameters
    other_parameters, # nuisance parameters
    bins=100,
    popsummary_path='results/popsummary/',
    datadir='data',
    metadata_label="bbh", # doesn't exist, will skip metadata saving
    overwrite=True,
    minima={'log_mass_1': np.log(mmin)},
    maxima={},
    parametric_models={},
    hyperparameters={},
    lower_triangular=False,
    priors=priors,
    )