import numpy as np
from jax import jit, numpy as jnp
import popsummary
import os
import json
from functools import reduce
from scipy.special import logsumexp as LSE
from tqdm import tqdm
from ..models.car import axes_tril
from .validate import validate_pixelpop_inference
from .post_processing import *
import xarray as xr
from ..models.gwpop_models import map_to_gwpop_parameters

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


def save_text_summary(
        rhat_results, ess_results, error_stats, rhat_threshold=1.01, ess_threshold=100, 
        filename='validation_summary.txt'
        ):
    
    with open(filename, 'w') as f:
        f.write("PIXELPOP INFERENCE VALIDATION SUMMARY\n")
        f.write("="*40 + "\n\n")

        all_rhat = np.concatenate([v.values.flatten() for v in rhat_results.data_vars.values()])
        all_ess = np.concatenate([v.values.flatten() for k, v in ess_results.data_vars.items() if '_bulk' in k])
        
        rhat_fail_pct = (all_rhat > rhat_threshold).mean() * 100
        ess_fail_pct = (all_ess < ess_threshold).mean() * 100

        f.write("--- Global Statistics ---\n")
        f.write(f"R-hat:  Median = {np.nanmedian(all_rhat):.4f}, Max = {np.nanmax(all_rhat):.4f}\n")
        f.write(f"ESS:    Median = {np.nanmedian(all_ess):.1f}, Min = {np.nanmin(all_ess):.1f}\n")
        f.write(f"Failures: {rhat_fail_pct:.2f}% exceed R-hat {rhat_threshold}\n")
        f.write(f"          {ess_fail_pct:.2f}% fall below ESS {ess_threshold}\n\n")

        # Identify worst parameters
        f.write("--- Worst Offenders (Highest R-hat) ---\n")
        f.write(f"{'Parameter/Index':<30} | {'R-hat':<10}\n")
        f.write("-" * 45 + "\n")
        
        # Create a list of all parameter names and their values
        rhat_list = []
        for var in rhat_results.data_vars:
            vals = rhat_results[var].values.flatten()
            if vals.size == 1:
                rhat_list.append((var, vals[0]))
            else:
                # For high-dim, find the max in that variable
                idx = np.nanargmax(vals)
                rhat_list.append((f"{var}[flat_idx {idx}]", vals[idx]))
        
        # Sort by R-hat descending and take top 5
        worst_rhats = sorted(rhat_list, key=lambda x: x[1], reverse=True)[:5]
        for name, val in worst_rhats:
            f.write(f"{name:<30} | {val:<10.4f}\n")
        f.write("\n")

        f.write("--- Monte Carlo Systematics ---\n")
        for k, v in error_stats.items():
            f.write(f"{k:45}: {v:.6f}\n")
        f.write("\n")

        f.write("--- Low-Dimensional Parameter Details ---\n")
        f.write(f"{'Parameter':<20} | {'R-hat':<10} | {'ESS (Bulk)':<10}\n")
        f.write("-" * 45 + "\n")
        
        for var in rhat_results.data_vars:
            if rhat_results[var].size < 5:
                rh = rhat_results[var].values.flatten()[0]
                es = ess_results[f"{var}_bulk"].values.flatten()[0]
                f.write(f"{var:<20} | {rh:<10.4f} | {es:<10.1f}\n")

    print(f"Validation summary saved to {filename}")


def create_popsummary(
        pixelpop_data, hyperposterior_chains, run_name="", popsummary_path='../results/popsummary/',
        datadir='../data', metadata_label="", overwrite=False,  
        ):
    '''
    Parameters
    ----------
    pixelpop_data: PixelPopData
        The data object containing posteriors, injections, and model settings.
    hyperposterior_chains : list[dict] or dict
        Either a list of dictionaries of the hyperposterior samples [np.NDarray 
        shaped as (Nsamples,...)], indexing the independent chains, or single 
        dictionary of one chain
    run_name : str
        name for popsummary file, defaults to PixelPopData name
    popsummary_path : str
        relative path from script directory where to save the popsummary file
    datadir : str
        relative path from the script directory where the gwpopulation_pipe 
        data (posterior samples, injections, etc.) was stored
    metadata_label : str
        additional subdirectory of datadir where the gwpopulation_pipe data is 
        stored, e.g., datadir/metadata_label/ contains gwpopulation_pipe metadata
    overwrite : bool
        whether to overwrite existing popsummary data
    '''

    pixelpop_parameters = pixelpop_data.pixelpop_parameters
    other_parameters = pixelpop_data.other_parameters
    bins = pixelpop_data.bins
    priors = pixelpop_data.priors
    parametric_models = pixelpop_data.parametric_models
    parameter_to_hyperparameters = pixelpop_data.parameter_to_hyperparameters
    dimension = pixelpop_data.dimension
    lower_triangular = pixelpop_data.lower_triangular
    minima = pixelpop_data.minima
    maxima = pixelpop_data.maxima
    
    if run_name == "":
        run_name = pixelpop_data.name
        
    parameters = pixelpop_parameters + other_parameters
    gwparameters = []
    for p in parameters:
        if p in map_to_gwpop_parameters:
            gwparameters += map_to_gwpop_parameters[p]
        else:
            gwparameters += [p]

    if type(hyperposterior_chains) == list:
        if len(hyperposterior_chains) == 1:
            hyperposterior = hyperposterior_chains[0]
        else:
            hyperposterior = reduce(combine_chains, hyperposterior_chains)
    else:
        assert isinstance(hyperposterior_chains, dict)
        hyperposterior = hyperposterior_chains 

    hyperposterior, Nsamples = pixelpop_data.fill_out_hyperposterior(hyperposterior)
    
    popsummary_path = os.path.join(popsummary_path, run_name)
    if not os.path.exists(popsummary_path):
        os.makedirs(popsummary_path)
    popsummary_filepath = os.path.join(popsummary_path, run_name + '_popsummary.h5')

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
        event_parameters=gwparameters
        )
    # set one dimensional rates
    skip_parameters = []
    pp_grids = pixelpop_data.bin_axes
    
    if lower_triangular:
        # first two axes are assumed lower triangular
        assert bins[0] == bins[1]
        skip_parameters = pixelpop_parameters[:2]
        axes = tuple(range(2, dimension))
        # 
        ldm1 = pixelpop_data.logdV[0]
        ldm2 = pixelpop_data.logdV[1]

        # log of the volume element for all other parameters
        lda = pixelpop_data.logdV[2:]
        
        hyperposterior['log_rate'] = np.array(
                [LSE(Rsub) + np.sum(pixelpop_data.logdV) for Rsub in tqdm(hyperposterior['merger_rate_density'], desc=f'Computing integrated log rates')]
            ) - np.log(2) # divide by 2 bc lower triangular
        hyperposterior['merger_rate_density'] = np.log(axes_tril(np.exp(hyperposterior['merger_rate_density']), axes=(1,2)))
        R = hyperposterior['merger_rate_density']
        # converting to comoving merger rate density, if applicable
        if 'redshift' not in pixelpop_parameters: 
            # redshift marginalization requires cosmological factors
            m1 = np.array(
                [LSE(Rsub, axis=(1,)+axes) + np.sum(lda) + ldm2 for Rsub in tqdm(R, desc=f'Computing mass_1 marginals')]
            )
            m2 = np.array(
                [LSE(Rsub, axis=(0,)+axes) + np.sum(lda) + ldm1 for Rsub in tqdm(R, desc=f'Computing mass_2 marginals')]
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
    R = hyperposterior['merger_rate_density']

    if pixelpop_data.has_window:
        print('Calculating window factors for joint pixelpop rate')
        # build full meshgrid so window functions that depend on multiple
        # pixelpop parameters (e.g. m2 = m1*q) have all values available
        bin_centers = [0.5 * (g[:-1] + g[1:]) for g in pp_grids]
        grids = np.meshgrid(*bin_centers, indexing='ij')
        grid_data = {par: grids[jj] for jj, par in enumerate(pixelpop_parameters)}

        window_in_bins = np.zeros(R.shape)
        for par in pixelpop_data.window_parameters:
            assert 'log_marginal_' + par in hyperposterior

            # evaluate window on the full N-D grid of bin centers
            window_factors = np.array([parametric_models[par+'_window'](
                    grid_data,
                    *[hyperposterior[k][ii] for k in parameter_to_hyperparameters[par+'_window']]
                    )
                for ii in range(Nsamples)])
            window_in_bins += window_factors

        # Use a local variable for the windowed rate — do NOT mutate
        # hyperposterior['merger_rate_density'], which is the raw ICAR
        # field and still needed by reweight_events_and_injections
        # (where the window is applied per-event via the parametric model).
        R_windowed = R + window_in_bins
        log_norms = LSE(R_windowed, axis=tuple(range(1, R_windowed.ndim))) + np.sum(pixelpop_data.logdV)
        hyperposterior['log_rate'] = log_norms
        for par in pixelpop_parameters:
            par_idx = pixelpop_parameters.index(par)
            par_axis = par_idx + 1
            sum_axes = tuple(ax for ax in range(1, R_windowed.ndim) if ax != par_axis)
            logdV_other = np.sum(pixelpop_data.logdV[:par_idx]) + np.sum(pixelpop_data.logdV[par_idx+1:])
            assert 'log_marginal_' + par in hyperposterior
            hyperposterior['log_marginal_' + par] = LSE(R_windowed, axis=sum_axes) + logdV_other - log_norms[:,None]

        R = R_windowed

    # Save hyperparameter samples after the window block has corrected log_rate
    result.set_hyperparameter_samples(np.array([hyperposterior[h] for h in h_keys]).T, overwrite=overwrite)

    lrs = hyperposterior['log_rate']
    
    for ii, par in enumerate(parameters):
        print(f'Saving {par} rates on grids...')
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
            pos = np.linspace(minima[par], maxima[par], bins[ii]+1)
            
        else:
            try:
                # for joint models where the marginal is known, e.g., IID spins which are treated as a joint distribution
                # or spin tilts where the marginal is the same for cos_tilt_1 and cos_tilt_2, the parametric model should
                # be constructed so that if it is called on a dataset without trailing _1 or _2, it returns the marginal
                pos = jnp.linspace(minima[par], maxima[par], 1000)
                try:
                    rate_func = jit(parametric_models[par])
                except:
                    rate_func = parametric_models[par]
                required_keys = parameter_to_hyperparameters[par]
                rates = np.array([rate_func({par.replace('_window', ''): pos}, *[hyperposterior[k][jj] for k in required_keys]) for jj in tqdm(range(Nsamples))])
                if 'redshift' not in pixelpop_parameters:
                    rates += lrs[:,None]
                    # if redshift is one of the pp parameters, the log rate is a naive average over redshift which does 
                    # not account for cosmo term. In this case, we just save the probability density instead.
            except Exception as e:
                print(f'Could not save {par} rates on grids with exception\n{e}\n, skipping...')
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
    result.set_rates_on_grids(
        grid_key='joint_pixelpop_rate',
        grid_params=pixelpop_parameters,
        positions=pos,
        rates=np.exp(R.reshape(Nsamples,np.prod(bins))),
        overwrite=overwrite
    )

    # validation of results
    rhat_results, ess_results, error_stats, summary = validate_pixelpop_inference(
        hyperposterior_chains,
        pixelpop_data,
        )
    for k in summary:
        result.set_metadata(k, summary[k], overwrite=overwrite)
    
    try:
        from .. import __version__
    except ImportError:
        __version__ = "unknown"
    result.set_metadata('pixelpop_version', __version__, overwrite=overwrite)
    # save details about the validation to a validation_statistics h5 file
    validation_filename = os.path.join(popsummary_path, 'validation_statistics.h5')

    error_ds = xr.Dataset(error_stats)
    rhat_results.to_netcdf(validation_filename, group='rhat', engine='h5netcdf')
    ess_results.to_netcdf(validation_filename, group='ess', engine='h5netcdf', mode='a')
    error_ds.to_netcdf(validation_filename, group='systematics', engine='h5netcdf', mode='a')

    summary_filepath = os.path.join(popsummary_path, 'validation_summary.txt')
    
    save_text_summary(
        rhat_results, 
        ess_results, 
        error_stats, 
        rhat_threshold=1.01, 
        ess_threshold=100, 
        filename=summary_filepath
        )
    
    # reweight events and injections
    reweight_events_and_injections(result, hyperposterior, pixelpop_data, overwrite=overwrite)
