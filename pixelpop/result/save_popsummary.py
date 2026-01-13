import h5ify
import pickle as pkl
import numpy as np
from jax import jit, numpy as jnp
import popsummary
import os
from glob import glob
from functools import reduce
import json
from ..models import gwpop_models
from scipy.special import logsumexp as LSE
from tqdm import tqdm
from ..models.car import axes_tril
from .validate import *

def combine_chains(chain_1, chain_2):
    for k in chain_1.keys():
        assert k in chain_2.keys()
    assert len(chain_1.keys()) == len(chain_2.keys())

    x = {}
    for k in chain_1.keys():
        x[k] = np.concatenate((chain_1[k], chain_2[k]), axis=0)
    return x

def get_posterior(rundir, chain_regex='chain_*_samples', result_file_type='h5'):
    fpath = os.path.join(rundir, chain_regex+'.'+result_file_type)
    paths = glob(fpath)
    print(f"I got {len(paths)} unique chains")
    chains = []
    for p in paths:
        if result_file_type == 'h5':
            chain = h5ify.load(p)
        elif result_file_type == 'pkl':
            with open(p, 'rb') as ff:
                chain = pkl.load(ff)
        else:
            print('h5 and pkl are only accepted result_file_type')
            return
        chains.append(chain)
    combined_chains = reduce(combine_chains, chains)
    for k in combined_chains:
        print(k, combined_chains[k].shape)

    return combined_chains

def get_input_metadata(file_label, datadir='../data'):
    file_path = os.path.join(datadir, file_label, 'event_data.json')
    try:
        with open(file_path, 'r') as file:
            metadata = json.load(file)
    except FileNotFoundError:
        file_path = os.path.join(datadir, file_label, 'data', 'event_data.json')
        with open(file_path, 'r') as file:
            metadata = json.load(file)
    wf_paths = metadata.keys()
    wfs = []
    for p in wf_paths:
        if 'S230529' in p or 'GW230529' in p:
            wfs.append('GW230529')
            continue
        name = p.split('/')[-1]
        if name.startswith('S'):
            wfs.append(name.split('-')[0])
        else:
            wfs.append(name.split('-')[3].replace('_PEDataRelease_mixed_cosmo.h5', '').replace('_PEDataRelease_cosmo.h5', ''))
    # print(wfs)
    return wfs, wf_paths, metadata

def get_analysis_metadata(
        hyperposterior, pixelpop_data, verbose=True
        ):
    pass
    

def create_popsummary(
        pixelpop_data, hyperposterior, run_name, popsummary_path='../results/popsummary/',
        datadir='../data', metadata_label="", overwrite=False,  
        ):
    '''
    Parameters
    ----------
    pixelpop_data: 
        TODO
    hyperposterior: dict
        A dictionary of the hyperposterior samples [np.NDarray shaped as 
        (Nsamples,...)]
    run_name: str
        name for popsummary file
    popsummary_path: str
        relative path from script directory where to save the popsummary file
    datadir: str
        relative path from the script directory where the gwpopulation_pipe 
        data (posterior samples, injections, etc.) was stored
    metadata_label: str
        additional subdirectory of datadir where the gwpopulation_pipe data is 
        stored, e.g., datadir/metadata_label/ contains gwpopulation_pipe metadata
    overwrite: bool
        whether to overwrite existing popsummary data
    
    '''
    pixelpop_parameters = pixelpop_data.pixelpop_parameters
    other_parameters = pixelpop_data.other_parameters
    bins = pixelpop_data.bins
    priors = pixelpop_data.priors
    parametric_models = pixelpop_data.parametric_models
    parameter_to_hyperparameters = pixelpop_data.parameter_to_hyperparameters
    dimension = pixelpop_data.dimension
    
    parameters = pixelpop_parameters + other_parameters

    if type(hyperposterior) == list:
        if len(hyperposterior) == 1:
            hyperposterior = hyperposterior[0]
        else:
            hyperposterior = reduce(combine_chains, hyperposterior)

    delta_pars = {}
    for p in other_parameters:
        for h in parameter_to_hyperparameters[p]:
            if priors[h][1].__name__ == 'Delta':
                delta_pars[h] = priors[h][0][0]

    Nsamples = len(hyperposterior['log_likelihood'])
    for par in other_parameters:
        required_keys = parameter_to_hyperparameters[par]
        for k in required_keys:
            if not k in hyperposterior:
                hyperposterior[k] = delta_pars[k]*np.ones(Nsamples)

    if not os.path.exists(popsummary_path):
        os.makedirs(popsummary_path)
    popsummary_filepath = os.path.join(popsummary_path, run_name + '.h5')

    try:
        wfs, wf_paths, metadata = get_input_metadata(file_label=metadata_label, datadir=datadir)
    except Exception as e:
        print(f'Warning: {e}\nCould not load run metadata, skipping.')
        wfs, wf_paths, metadata = [], [], []
    h_keys = [x for x in hyperposterior.keys() if hyperposterior[x].ndim == 1]
    if not overwrite:
        if os.path.exists(popsummary_filepath):
            for counter in range(100):
                new_name = os.path.join(popsummary_path, 'old_' + f'{counter}_' + run_name + '.h5')
            os.rename(popsummary_filepath, new_name)
    result = popsummary.popresult.PopulationResult(
        popsummary_filepath,
        hyperparameters=h_keys,
        events=wfs,
        event_waveforms=[metadata[p]['waveform'] for p in wf_paths],
        event_sample_IDs=[metadata[p]['label'] for p in wf_paths],
        event_parameters=parameters
        )
    result.set_hyperparameter_samples(np.array([hyperposterior[h] for h in h_keys]).T, overwrite=overwrite)

    # set one dimensional rates
    skip_parameters = []
    pp_grids = pixelpop_data.bin_axes

    if lower_triangular:
        # first two axes are assumed lower triangular
        assert bins[0] == bins[1]
        skip_parameters = pixelpop_parameters[:2]
        axes = tuple(range(2, len(pixelpop_parameters)))
        # 
        ldm1 = pixelpop_data.log_dV[0]
        ldm2 = pixelpop_data.log_dV[1]

        # log of the volume element for all other parameters
        lda = pixelpop_data.log_dV[2:]

        hyperposterior['log_rate'] = LSE(hyperposterior['merger_rate_density']) + np.sum(pixelpop_data.log_dV) - np.log(2) # divide by 2 bc lower triangular
        hyperposterior['merger_rate_density'] = np.log(axes_tril(np.exp(hyperposterior['merger_rate_density']), axes=(1,2)))
        
        # converting to comoving merger rate density, if applicable
        R = hyperposterior['merger_rate_density']
        if 'redshift' not in pixelpop_parameters: 
            # redshift marginalization requires cosmological factors
            m1 = np.array(
                [LSE(Rsub, axis=(2,)+axes) + np.sum(lda) + ldm2 for Rsub in tqdm(R, desc=f'Computing mass_1 marginals')]
            )
            m2 = np.array(
                [LSE(Rsub, axis=(1,)+axes) + np.sum(lda) + ldm1 for Rsub in tqdm(R, desc=f'Computing mass_2 marginals')]
            )
            
            m1 = np.concatenate((m1[:,0][:,None], m1), axis=1)
            m2 = np.concatenate((m2[:,0][:,None], m2), axis=1)
            
            result.set_rates_on_grids(
                grid_key=pixelpop_parameters[0],
                grid_params=pixelpop_parameters[0],
                positions=pp_grids[0],
                rates=np.exp(m1),
                overwrite=overwrite
                )
            result.set_rates_on_grids(
                grid_key=pixelpop_parameters[1],
                grid_params=pixelpop_parameters[1],
                positions=pp_grids[1],
                rates=np.exp(m2),
                overwrite=overwrite
                )
            
        for ii_par, par in enumerate(pixelpop_parameters[2:]):
            sum_axes = (0,1) + axes[:ii_par] + axes[ii_par+1:]
            total_d = ldm1 + ldm2 + np.sum(lda[:ii_par]) + np.sum(lda[ii_par+1:])
            marginal = np.array(
                [LSE(Rsub, axis=sum_axes) + total_d for Rsub in tqdm(R, desc=f'Computing {par} marginals')]
            )
            hyperposterior[f'log_marginal_{par}'] = marginal
    
    assert 'log_rate' in hyperposterior
    lrs = hyperposterior['log_rate']
    
    for ii, par in enumerate(parameters):
        if par in pixelpop_parameters:
            if par in skip_parameters:
                continue
            
            if 'redshift' in pixelpop_parameters:
                if par != 'redshift':
                    # naive marginalization over redshift neglects implicit dVc/dz 1/1+z term
                    continue
            assert 'log_marginal_' + par in hyperposterior
            rates = hyperposterior['log_marginal_'+par]
            rates = np.concatenate((rates[:,0][:,None], rates), axis=1)
            rates += lrs[:,None]
            
        else:
            try:
                print(f'Saving {par} rates on grids...')
                pos = jnp.linspace(minima[par], maxima[par], 1000)
                try:
                    rate_func = jit(parametric_models[par])
                except:
                    rate_func = parametric_models[par]
                required_keys = parameter_to_hyperparameters[par]
                rates = np.array([rate_func({par: pos}, *[hyperposterior[k][ii] for k in required_keys]) for jj in tqdm(range(Nsamples))])
                rates += lrs[:,None]
            except:
                print(f'Could not save {par} rates on grids, skipping...')
                continue
            
        result.set_rates_on_grids(
            grid_key=par,
            grid_params=par,
            positions=pos,
            rates=np.exp(rates),
            overwrite=overwrite
            )
    
    # Nd stuff
    x_axes = np.meshgrid(*pp_grids, indexing='ij')
    pos = np.vstack([
        x.flatten() for x in x_axes
        ])
    nd_pp = hyperposterior['merger_rate_density']
    result.set_rates_on_grids(
        grid_key='joint_pixelpop_rate',
        grid_params=pixelpop_parameters,
        positions=pos,
        rates=np.exp(nd_pp.reshape(Nsamples,np.prod(bins))),
        overwrite=overwrite
    )

    # validate hyperposterior metrics: error statistic, neffs, and rhats
    # ....
    # ....
    # ....
    # ....
