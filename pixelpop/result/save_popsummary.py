import h5ify
import pickle as pkl
import numpy as np
import popsummary
import os
from glob import glob
from functools import reduce
import json
from ..models import gwpop_models
from scipy.special import logsumexp as LSE

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

def get_run_metadata(file_label, datadir='../data'):
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

def create_popsummary(
        posterior, run_name, pixelpop_parameters, other_parameters, bins=100, popsummary_path='../results/popsummary/', 
        datadir='../data', metadata_label="", overwrite=False, minima={}, maxima={}, parametric_models={}, hyperparameters={}, 
        priors={}, lower_triangular=False, 
        ):
    '''
    Parameters
    ----------
    posterior: dict
        A dictionary of the hyperposterior samples [np.NDarray shaped as 
        (Nsamples,...)]
    run_name: str
        name for popsummary file
    pixelpop_parameters: list
        list of strings, containing the parameters over which the pixelpop model
        was defined    
    other_parameters: list
        list of strings, containing the parameters for the other parameters 
        necessary in the population model
    bins: int (TODO: or list)
        number of bins along each axis
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
    minima: dict
        dictionary of gwparameter keys (mass_1, mass_ratio, chi_eff, etc.) to the 
        minimum value in the space. If no key is passed, defaults to typical bbh 
        values, e.g., mass_1: 3, mass_ratio: 0., chi_eff: -1
    maxima: dict
        dictionary of gwparameter keys (mass_1, mass_ratio, chi_eff, etc.) to the 
        maximum value in the space. If no key is passed, defaults to typical bbh 
        values, e.g., mass_1: 200, mass_ratio: 1., chi_eff: 1
    parametric_models: dict    
        dictionary of gwparameter keys (mass_1, mass_ratio, chi_eff, etc.) to 
        parametric model function. If no key is passed, defaults to GWTC-3 default 
        parametric models
    hyperparameters: dict
        dictionary of gwparameter keys (mass_1, mass_ratio, chi_eff, etc.) to a list 
        of the (string) hyperparameter names for the corresponding parametric 
        function
    priors: dict
        dictionary of hyperparmeter name to a tuple containing (zeroth index) the 
        list of arguments for the numpyro distribution from which to sample the 
        hyperparameter, and (first index) the numpyro distribution, e.g., 
        'max_z': ([2.4], numpyro.distributions.Delta)
    lower_triangular: bool
        whether to use the lower_triangular formalism where p1 > p2 is assumed.
        Usually this will only be used when joint mass_1, mass_2 inference is done 
    '''
    
    if type(posterior) == list:
        if len(posterior) == 1:
            posterior = posterior[0]
        else:
            posterior = reduce(combine_chains, posterior)

    parameter_to_hyperparameters = gwpop_models.gwparameter_to_hyperparameters.copy()
    parameter_to_hyperparameters.update(hyperparameters)

    # hyperparameters_plausible = gwpop_models.typical_hyperparameters.copy()

    parameter_to_gwpop_model = {}
    for p in other_parameters:
        if p in parametric_models:
            parameter_to_gwpop_model[p] = parametric_models[p]
        else:
            parameter_to_gwpop_model[p] = gwpop_models.gwparameter_to_model[p]
    
    hyperparameter_priors = {}
    delta_pars = {}
    for p in other_parameters:
        for h in parameter_to_hyperparameters[p]:
            if h in priors:    
                pprint = priors[h]
                print(f'Using custom prior {h} = {pprint[1].__name__}({str(pprint[0])[1:-1]}) in {p} model')
                hyperparameter_priors[h] = priors[h]
            else:
                pprint = gwpop_models.default_priors[h]
                print(f'Using default prior {h} = {pprint[1].__name__}({str(pprint[0])[1:-1]}) in {p} model')
                hyperparameter_priors[h] = gwpop_models.default_priors[h]
            if hyperparameter_priors[h][1].__name__ == 'Delta':
                delta_pars[h] = hyperparameter_priors[h][0][0]

    Nsamples = len(posterior['log_likelihood'])
    for par in other_parameters:
        required_keys = parameter_to_hyperparameters[par]
        for k in required_keys:
            if not k in posterior:
                posterior[k] = delta_pars[k]*np.ones(Nsamples)

    if not os.path.exists(popsummary_path):
        os.makedirs(popsummary_path)
    popsummary_filepath = os.path.join(popsummary_path, run_name + '.h5')
    if not 'redshift' in pixelpop_parameters:
        log_norms = np.array([
            parameter_to_gwpop_model['redshift'](
                None, 
                *[posterior[h][ii] for h in parameter_to_hyperparameters['redshift']],
                return_normalization=True
                )
            for ii in range(Nsamples)])
    else:
        log_norms = np.zeros_like(Nsamples)
    
    parameters = pixelpop_parameters + other_parameters
    try:
        wfs, wf_paths, metadata = get_run_metadata(file_label=metadata_label, datadir=datadir)
    except Exception as e:
        print(f'Warning: {e}\nCould not load run metadata, skipping.')
        wfs, wf_paths, metadata = [], [], []
    h_keys = [x for x in posterior.keys() if posterior[x].ndim == 1]
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
    result.set_hyperparameter_samples(np.array([posterior[h] for h in h_keys]).T, overwrite=overwrite)

    minima.update({k: gwpop_models.bbh_minima[k] for k in gwpop_models.bbh_minima if k not in minima})
    maxima.update({k: gwpop_models.bbh_maxima[k] for k in gwpop_models.bbh_maxima if k not in maxima})

    # set one dimensional rates
    skip_parameters = []
    pp_grids = []

    if lower_triangular:
        # do something special

        skip_parameters = pixelpop_parameters[:2]
        axes = tuple(range(3, len(pixelpop_parameters)+1))

        pp_grids.append(np.linspace(minima[pixelpop_parameters[0]], maxima[pixelpop_parameters[0]], bins+1))
        pp_grids.append(np.linspace(minima[pixelpop_parameters[1]], maxima[pixelpop_parameters[1]], bins+1))
        # assert posterior['merger_rate_density'].ndim == 3 # assure only two parameters FOR NOW

        dm1 = (maxima[pixelpop_parameters[0]] - minima[pixelpop_parameters[0]]) / bins
        dm2 = (maxima[pixelpop_parameters[1]] - minima[pixelpop_parameters[1]]) / bins
        lda = np.log(np.array([
            (maxima[pixelpop_parameters[ii]] - minima[pixelpop_parameters[ii]]) / bins for ii in range(2, len(pixelpop_parameters))
            ]))

        posterior['log_rate'] = LSE(posterior['merger_rate_density']) + np.log(dm1) + np.log(dm2) + np.sum(lda) - np.log(2) # divide by 2
        posterior['merger_rate_density'] = np.log(np.tril(np.exp(posterior['merger_rate_density'])))
        R = posterior['merger_rate_density'] - np.expand_dims(log_norms, axis=(1,2)+axes)
        m1, m2 = LSE(R, axis=(2,)+axes) + np.log(dm2) + np.sum(lda), LSE(R, axis=(1,)+axes) + np.log(dm1) + np.sum(lda)
    
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
            sum_axes = (1,2) + axes[:ii_par] + axes[ii_par+1:]
            total_d = np.log(dm1) + np.log(dm2) + np.sum(lda[:ii_par]) + np.sum(lda[ii_par+1:])
            marginal = LSE(R, axis=sum_axes) + total_d - np.log(2)
            posterior[f'log_marginal_{par}'] = marginal

    assert 'log_rate' in posterior
    lrs = posterior['log_rate'] - log_norms
    
    for par in parameters:
        if par in pixelpop_parameters:
            if par in skip_parameters:
                continue
            pos = np.linspace(minima[par], maxima[par], bins+1)
            pp_grids.append(pos)
            if 'redshift' in pixelpop_parameters:
                if par != 'redshift':
                    # naive marginalization over redshift neglects implicit dVc/dz 1/1+z term
                    continue
            assert 'log_marginal_' + par in posterior
            rates = posterior['log_marginal_'+par]
            rates = np.concatenate((rates[:,0][:,None], rates), axis=1)
            rates += lrs[:,None]
            print(par, rates.shape)
            # to use in plt.step plots
        else:
            assert par in other_parameters # I mean... come on, obviously but OK
            try:
                pos = np.linspace(minima[par], maxima[par], 1000)
                rate_func = parameter_to_gwpop_model[par]
                required_keys = parameter_to_hyperparameters[par]
                rates = np.array([rate_func({par: pos}, *[posterior[k][ii] for k in required_keys]) for ii in range(Nsamples)])
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
    nd_pp = posterior['merger_rate_density'] - np.expand_dims(log_norms, axis=tuple(range(1, len(pixelpop_parameters)+1)))
    result.set_rates_on_grids(
        grid_key='joint_pixelpop_rate',
        grid_params=pixelpop_parameters,
        positions=pos,
        rates=np.exp(nd_pp.reshape(Nsamples,bins**len(pixelpop_parameters))),
        overwrite=overwrite
    )
