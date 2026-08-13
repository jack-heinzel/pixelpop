"""
Tests for the dense mass-matrix specification helpers.

``resolve_dense_mass`` turns a human-readable spec into the list of tuples
numpyro expects, so most of these check the expansion rules. The last few run
NUTS to confirm the blocks numpyro is handed actually adapt.
"""
from types import SimpleNamespace

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer import MCMC, NUTS

from pixelpop.models.gwtc_defaults import gwtc6_default
from pixelpop.models.reparameterization import reparameterized_sites
from pixelpop.models.probabilistic import (
    get_latent_sites,
    parametric_dense_blocks,
    resolve_dense_mass,
    trace_model,
)

PARAMETERS = ['log_mass_1', 'mass_ratio', 'a', 't', 'redshift']


@pytest.fixture
def pixelpop_data():
    """The parts of PixelPopData that resolve_dense_mass reads."""
    return SimpleNamespace(
        other_parameters=list(PARAMETERS),
        parameter_to_hyperparameters={
            p: list(gwtc6_default.hyperparameters[p]) for p in PARAMETERS
        },
        priors=dict(gwtc6_default.priors),
    )


# ---------------------------------------------------------------------------
# Expansion rules
# ---------------------------------------------------------------------------

def test_bools_and_none_pass_through(pixelpop_data):
    assert resolve_dense_mass(True, pixelpop_data) is True
    assert resolve_dense_mass(False, pixelpop_data) is False
    assert resolve_dense_mass(None, pixelpop_data) is False


def test_parametric_gives_one_block_per_model(pixelpop_data):
    sites = reparameterized_sites(pixelpop_data.priors)
    blocks = resolve_dense_mass('parametric', pixelpop_data)
    assert len(blocks) == len(PARAMETERS)
    seen = set()
    for block, parameter in zip(blocks, PARAMETERS):
        expected = []
        for h in pixelpop_data.parameter_to_hyperparameters[parameter]:
            if pixelpop_data.priors[h][1] is dist.Delta:
                continue
            # numpyro needs the blocks to partition the latent sites, so a site
            # already claimed by an earlier block is dropped rather than repeated.
            # mlow_1 and mlow_2 share one site and sit in different models, so
            # this is not hypothetical.
            site = sites.get(h, h)
            if site not in seen:
                seen.add(site)
                expected.append(site)
        assert list(block) == expected


def test_ordered_pairs_block_under_their_sampled_name(pixelpop_data):
    """mlow_1/mlow_2 are deterministics of the site NUTS actually adapts, so a
    block naming them has to come back holding that site -- otherwise it is
    dropped as unsampled and the correlations go unmodelled. Both members share
    one length-2 site, so the block is a single name and the mass matrix picks up
    their correlation as that site's own 2x2 block."""
    site = reparameterized_sites(pixelpop_data.priors)['mlow_1']
    blocks = resolve_dense_mass([('mlow_1', 'mlow_2')], pixelpop_data)
    assert blocks == [(site,)]

    flat = {h for block in resolve_dense_mass('parametric', pixelpop_data) for h in block}
    assert site in flat
    assert not {'mlow_1', 'mlow_2'} & flat


def test_parametric_drops_delta_priors(pixelpop_data):
    flat = {h for block in resolve_dense_mass('parametric', pixelpop_data) for h in block}
    # amax, mmax, gaussian_mass_maximum and max_z are fixed, not sampled, so NUTS
    # has no coordinate for them and numpyro would KeyError on the block.
    fixed = {h for h, prior in pixelpop_data.priors.items() if prior[1] is dist.Delta}
    assert fixed
    assert not (flat & fixed)


def test_parametric_needs_pixelpop_data():
    with pytest.raises(ValueError, match='pixelpop_data'):
        resolve_dense_mass('parametric')


def test_model_dimension_names_expand_to_their_hyperparameters(pixelpop_data):
    blocks = resolve_dense_mass(
        [('log_mass_1', 'mass_ratio', 'redshift'), ('a', 't')], pixelpop_data
    )
    assert len(blocks) == 2
    masses, spins = blocks
    assert 'alpha_1' in masses and 'beta' in masses and 'lamb' in masses
    assert 'mu_1_chi' in spins and 'xi_spin' in spins
    assert not set(masses) & set(spins)


def test_hyperparameters_and_model_names_mix_in_one_block(pixelpop_data):
    blocks = resolve_dense_mass([('redshift', 'beta')], pixelpop_data)
    assert list(blocks[0]) == ['lamb', 'beta']


def test_bare_string_element_is_a_single_site_block(pixelpop_data):
    assert resolve_dense_mass(['lnsigma'], pixelpop_data) == [('lnsigma',)]


def test_parametric_expands_in_place_among_explicit_blocks(pixelpop_data):
    nonparametric = ('log_nu_spde', 'log_ranges', 'lnsigma')
    blocks = resolve_dense_mass(['parametric', nonparametric], pixelpop_data)
    assert blocks[-1] == nonparametric
    assert blocks[:-1] == resolve_dense_mass('parametric', pixelpop_data)


def test_repeated_site_stays_only_in_the_first_block(pixelpop_data):
    """numpyro asserts the blocks partition the latent sites, so a hyperparameter
    shared by two models -- or named by hand and then again by 'parametric' --
    must appear exactly once."""
    blocks = resolve_dense_mass([('beta',), 'parametric'], pixelpop_data)
    flat = [h for block in blocks for h in block]
    assert flat.count('beta') == 1
    assert blocks[0] == ('beta',)


def test_unsampled_sites_are_dropped_rather_than_raising(pixelpop_data):
    blocks = resolve_dense_mass(
        [('lamb', 'not_a_site'), ('log_nu_spde',)], pixelpop_data,
        latent_sites={'lamb', 'alpha_1'},
    )
    assert blocks == [('lamb',)]


def test_everything_dropped_falls_back_to_diagonal(pixelpop_data):
    assert resolve_dense_mass('parametric', pixelpop_data, latent_sites=set()) is False


def test_parametric_dense_blocks_keeps_declaration_order(pixelpop_data):
    blocks = parametric_dense_blocks(pixelpop_data)
    assert [b[0] for b in blocks] == [
        pixelpop_data.parameter_to_hyperparameters[p][0] for p in PARAMETERS
    ]


# ---------------------------------------------------------------------------
# Latent-site detection and NUTS
# ---------------------------------------------------------------------------

def _toy_model(scale=1.):
    x = numpyro.sample('x', dist.Normal(0., scale))
    y = numpyro.sample('y', dist.Normal(x, scale))
    numpyro.sample('z', dist.Normal(0., 1.), sample_shape=(3,))
    numpyro.deterministic('x_plus_y', x + y)
    numpyro.factor('penalty', -0.5 * x ** 2)
    numpyro.sample('obs', dist.Normal(y, 1.), obs=jnp.asarray(0.3))


def test_latent_sites_excludes_deterministic_factor_and_observed():
    trace = trace_model(_toy_model)
    assert get_latent_sites(trace) == {'x', 'y', 'z'}


def test_latent_sites_keeps_conditioned_initial_values():
    """trace_model conditions on initial_value, which marks those sites observed
    even though NUTS still samples them."""
    initial_value = {'x': jnp.asarray(0.5)}
    trace = trace_model(_toy_model, initial_value)
    assert get_latent_sites(trace, initial_value) == {'x', 'y', 'z'}
    assert 'x' not in get_latent_sites(trace)  # without the initial values it drops out


def test_resolved_blocks_are_accepted_by_nuts():
    """A dense block over a correlated pair adapts and samples without numpyro
    complaining that the blocks fail to partition the latent sites."""
    blocks = resolve_dense_mass(
        [('x', 'y')], latent_sites=get_latent_sites(trace_model(_toy_model))
    )
    mcmc = MCMC(NUTS(_toy_model, dense_mass=blocks), num_warmup=200,
                num_samples=200, progress_bar=False)
    mcmc.run(random.PRNGKey(0))

    inverse_mass_matrix = mcmc.last_state.adapt_state.inverse_mass_matrix
    assert set(inverse_mass_matrix) == {('x', 'y'), ('z',)}
    assert inverse_mass_matrix[('x', 'y')].shape == (2, 2)  # dense
    assert inverse_mass_matrix[('z',)].shape == (3,)        # diagonal remainder
    assert jnp.all(jnp.isfinite(mcmc.get_samples()['x']))
