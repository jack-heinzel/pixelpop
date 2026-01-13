import numpy as np
import jax.numpy as jnp
from arviz import rhat
from ..models import gwpop_models
from ..utils.data import place_in_bins

import population_error

class PixelPopRateFunction(object):
    def __init__(
        self, pixelpop_parameters, other_parameters, bins, dataset_bins, 
        parametric_models={}, hyperparameters={}
        # , priors={}, plausible_hyperparameters={}, UncertaintyCut=1., random_initialization=True, 
    ):
        self.pixelpop_parameters = pixelpop_parameters
        self.other_parameters = other_parameters
        if isinstance(bins, int):
            self.bins = (bins,) * len(pixelpop_parameters)
        elif isinstance(bins, (list, tuple)):
            self.bins = bins
        else:
            raise IOError(f"Bins is expected to be a tuple, list, or integer. {bins} is not a valid type ({type(bins)}).")
        
        parameter_to_hyperparameters = gwpop_models.gwparameter_to_hyperparameters.copy()
        parameter_to_hyperparameters.update(hyperparameters)

        parameter_to_gwpop_model = {}
        for p in other_parameters:
            if p in parametric_models:
                print(f'Updating {p} model from {gwpop_models.gwparameter_to_model[p].__name__} to {parametric_models[p].__name__}')
                print(f'\t ...with hyperparameters {parameter_to_hyperparameters[p]}')
                parameter_to_gwpop_model[p] = parametric_models[p]
            else:
                print(f'Using default {p} model {gwpop_models.gwparameter_to_model[p].__name__}')
                parameter_to_gwpop_model[p] = gwpop_models.gwparameter_to_model[p]

        self.parameter_to_hyperparameters = parameter_to_hyperparameters
        self.parameter_to_gwpop_model = parameter_to_gwpop_model

        self.dataset_bins = dataset_bins
        self.shape = dataset_bins[0].shape

    def __call__(self, dataset, hyperparameters):
        
        lp_parametric = self.log_prob_parametric_model(dataset, hyperparameters)
        lp_pixelpop = self.log_rate_pixelpop(dataset, hyperparameters)

        return jnp.exp(lp_parametric + lp_pixelpop)
    
    def log_prob_parametric_model(self, dataset, hyperparameters):
        
        log_probs = jnp.zeros(self.shape)
              
        for p in self.other_parameters:
            hs = self.parameter_to_hyperparameters[p] # hyperparameters appropriate for this model
            log_probs += self.parameter_to_gwpop_model[p](dataset, *[hyperparameters[h] for h in hs])
            
        return log_probs
    
    def cosmological_factor(self, dataset):

        if 'redshift' in self.pixelpop_parameters:
            from astropy import units
            max_z = jnp.maximum(jnp.max(dataset['redshift']), gwpop_models.bbh_maxima['redshift'])
            zs = jnp.linspace(1e-6, max_z, 10000)
            dVs = gwpop_models.COSMO.differential_comoving_volume(zs)
            if isinstance(dVs, units.quantity.Quantity):
                dVs = 4*jnp.pi*dVs.to(units.Gpc**3 / units.sr).value
            else:
                dVs = 4*jnp.pi* 1e-9 * dVs
            ln_dVTc = jnp.log(dVs) - jnp.log(1 + zs)
            dataset_zs = dataset['redshift']
            return jnp.array(jnp.interp(dataset_zs, zs, ln_dVTc))
            
        return jnp.zeros(self.shape)

    def log_rate_pixelpop(self, dataset, hyperparameters):

        ln_dVcdz = self.cosmological_factor(dataset)
        pp_rates = hyperparameters['merger_rate_density']
        return pp_rates[self.dataset_bins] + ln_dVcdz


def compute_error_statistics(
        hyperposterior, posteriors, injections, pixelpop_parameters, other_parameters, 
        bins, minima={}, maxima={}, parametric_models={}, hyperparameters={}, verbose=True
        ):

    event_bins, inj_bins, bin_axes, logdV, eprior, iprior \
        = place_in_bins(
            pixelpop_parameters, 
            posteriors, 
            injections, 
            bins=bins, 
            minima=minima, 
            maxima=maxima
            )

    elog_prior = posteriors.pop('log_prior') + eprior
    ilog_prior = injections.pop('log_prior') + iprior
    
    posteriors['prior'] = jnp.exp(elog_prior)
    injections['prior'] = jnp.exp(ilog_prior)
    
    event_pixelpop_model = PixelPopRateFunction(
        pixelpop_parameters,
        other_parameters,
        bins, 
        event_bins, 
        parametric_models,
        hyperparameters
    )

    injection_pixelpop_model = PixelPopRateFunction(
        pixelpop_parameters,
        other_parameters,
        bins, 
        inj_bins, 
        parametric_models,
        hyperparameters
    )

    # burn a call for each model
    first_hypersample = {k: hyperposterior[k][0] for k in hyperposterior.keys()}
    
    _ = event_pixelpop_model(posteriors, first_hypersample)
    _ = injection_pixelpop_model(injections, first_hypersample)
    
    error_dict = population_error.error_statistics(
        event_pixelpop_model, 
        injections, 
        posteriors, 
        hyperposterior, 
        vt_model_function=injection_pixelpop_model,
        include_likelihood_correction=True,
        rate=True,
        verbose=verbose,
        )
    
    return error_dict

def rank_normalized_rhat(hyperposterior):
    pass