"""
Per-catalog default model sets.

Each catalog generation fixes a different set of "strong" parametric models for
the marginalized (non-pixelated) parameters, together with the hyperparameters
they take and the priors on those hyperparameters. This module bundles each
generation into one :class:`CatalogDefaults`:

===================== =========================================================
``gwtc3_default``     O3: Power Law + Peak mass, smoothed power-law mass ratio,
                      Beta spin magnitude, isotropic + Gaussian tilt.
``gwtc4_default``     O4a: broken power law + two peaks mass, truncated normal
                      spin magnitude. Identical to the module-level registries
                      in :mod:`~pixelpop.models.gwpop_models`.
``gwtc5_default``     Same models as GWTC-4; kept as a separate object so its
                      priors can diverge without disturbing GWTC-4.
``gwtc6_default``     BBH: broken power law + two peaks mass, two-Gaussian
                      mixture component spins, free-mean tilt.
``gwtc6_fms_default`` Full mass spectrum: as GWTC-6 but with the twice-broken
                      power law mass model and the neutron-star spin cap, and
                      minimum masses opened down to 1 Msun.
===================== =========================================================

Usage
-----
The fields line up with what :class:`~pixelpop.utils.data.PixelPopData` accepts::

    from pixelpop.models import gwtc6_fms_default

    defaults = gwtc6_fms_default.merge(
        priors={'lamb': ([-2., 10.], dist.Uniform)},   # caller wins
    )
    PixelPopData(
        ...,
        parametric_models=defaults.models,
        parameter_to_hyperparameters=defaults.hyperparameters,
        priors=defaults.priors,
        minima=defaults.minima,
        maxima=defaults.maxima,
    )

Hyperparameter names are positional
-----------------------------------
``probabilistic.py`` calls each model as
``model(data, *[sample[h] for h in hyperparameters[parameter]])``, so the order
of each hyperparameter list **must** match the model's signature after ``data``.
Adding a name means passing that argument; the corresponding keyword default is
then ignored.

A note on ``mu_spin`` / ``sigma_spin``
--------------------------------------
GWTC-6 uses these names for the *tilt* mean and width (``$\\mu_t$``,
``$\\sigma_t$`` in its prior file), whereas GWTC-3/4/5 use ``mu_spin`` and
``var_spin`` for the spin *magnitude*. Each set carries a complete, self-
contained prior dict, so the collision is harmless as long as sets are not
mixed -- do not splice a GWTC-6 prior dict into a GWTC-4 model set.
"""
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import jax.numpy as jnp
from numpyro import distributions as dist

from . import gwpop_models as _gwpop
from .O4_models import (
    BrokenPowerlawPlusTwoPeaks_PrimaryMass,
    SmoothedPowerlaw_MassRatio,
    TripleBrokenPowerlawPlusTwoPeaks_PrimaryMass,
    two_gaussian_spin,
    two_gaussian_spin_fms,
)
from .gwpop_models import (
    PlanckWindow_MassRatio,
    PlanckWindow_PrimaryMass,
    PlanckWindow_SecondaryMass,
    PowerlawPlusPeak_MassRatio,
    PowerlawPlusPeak_PrimaryMass,
    PowerlawRedshiftPsi,
    chieff_gaussian,
    chip_gaussian,
    gwtc3_spin_default,
    iid_beta_spin,
    tilt_default,
    tilt_model,
)


@dataclass(frozen=True)
class CatalogDefaults:
    """
    The default parametric models, hyperparameters and priors for one catalog.

    Attributes
    ----------
    models : Mapping[str, callable]
        Population parameter -> model function, as ``gwparameter_to_model``.
    hyperparameters : Mapping[str, list of str]
        Population parameter -> ordered hyperparameter names. The order must
        match the model signature after ``data``.
    priors : Mapping[str, tuple]
        Hyperparameter name -> ``(args, numpyro distribution)``.
    minima, maxima : Mapping[str, float], optional
        Bin edges for the pixelated parameters, overriding ``bbh_minima`` /
        ``bbh_maxima``. Empty means "use the pixelpop BBH defaults".

    Notes
    -----
    The mappings are wrapped read-only and copied on construction, so a set can
    never be mutated through this object nor alias the module-level registries.
    Use :meth:`merge` to derive a variant.
    """

    models: MappingProxyType
    hyperparameters: MappingProxyType
    priors: MappingProxyType
    minima: MappingProxyType = field(default_factory=dict)
    maxima: MappingProxyType = field(default_factory=dict)

    def __post_init__(self):
        for name in ('models', 'hyperparameters', 'priors', 'minima', 'maxima'):
            object.__setattr__(
                self, name, MappingProxyType(dict(getattr(self, name)))
            )

    def merge(self, models=None, hyperparameters=None, priors=None,
              minima=None, maxima=None):
        """
        Return a new :class:`CatalogDefaults` with caller-supplied entries
        layered on top of these defaults.

        Mirrors the resolution order in ``PixelPopData.__post_init__``: a
        caller-supplied entry always wins, and anything it does not cover falls
        back to this set. Entries are merged key by key, so passing a partial
        dict overrides only those keys.

        Parameters
        ----------
        models, hyperparameters, priors, minima, maxima : Mapping, optional
            Overrides for the corresponding field.

        Returns
        -------
        CatalogDefaults
            A new set; ``self`` is unchanged.
        """
        updates = {}
        for name, override in (('models', models),
                               ('hyperparameters', hyperparameters),
                               ('priors', priors),
                               ('minima', minima),
                               ('maxima', maxima)):
            if override is not None:
                updates[name] = {**getattr(self, name), **dict(override)}
        return replace(self, **updates)


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

# Windows are shape modifiers on the merger rate rather than population models,
# and are identical across catalogs; only the hyperparameter names differ, since
# GWTC-6 splits the minimum mass into mlow_1 / mlow_2.
_WINDOW_MODELS = {
    'log_mass_1_window': PlanckWindow_PrimaryMass,
    'log_mass_2_window': PlanckWindow_SecondaryMass,
    'mass_ratio_window': PlanckWindow_MassRatio,
}

_EFFECTIVE_SPIN_MODELS = {
    'chi_eff': chieff_gaussian,
    'chi_p': chip_gaussian,
}
_EFFECTIVE_SPIN_HYPERS = {
    'chi_eff': ['mu_x', 'sig_x'],
    'chi_p': ['mu_xp', 'sig_xp'],
}


# ---------------------------------------------------------------------------
# GWTC-3 (O3)
# ---------------------------------------------------------------------------

gwtc3_default = CatalogDefaults(
    models={
        'mass_1': PowerlawPlusPeak_PrimaryMass,
        'log_mass_1': PowerlawPlusPeak_PrimaryMass,
        'mass_ratio': PowerlawPlusPeak_MassRatio,
        'redshift': PowerlawRedshiftPsi,
        'a': iid_beta_spin,
        't': tilt_default,
        'spin': gwtc3_spin_default,
        **_EFFECTIVE_SPIN_MODELS,
        **_WINDOW_MODELS,
    },
    hyperparameters={
        # PowerlawPlusPeak_PrimaryMass(data, alpha, minimum, maximum, delta_m,
        #                              mpp, sigpp, lam)
        'mass_1': ['alpha', 'mmin', 'mmax', 'delta_m', 'mpp', 'sigpp', 'lam'],
        'log_mass_1': ['alpha', 'mmin', 'mmax', 'delta_m', 'mpp', 'sigpp', 'lam'],
        # PowerlawPlusPeak_MassRatio(data, slope, minimum, delta_m)
        'mass_ratio': ['beta', 'mmin', 'delta_m'],
        'redshift': ['lamb', 'max_z'],
        'a': ['mu_spin', 'var_spin'],            # iid_beta_spin(data, mu, var)
        't': ['sig_tilt', 'zeta_tilt'],          # tilt_default(data, sig, zeta)
        'spin': ['mu_spin', 'var_spin', 'sig_tilt', 'zeta_tilt'],
        'log_mass_1_window': ['mmin', 'delta_m'],
        'log_mass_2_window': ['mmin', 'delta_m'],
        'mass_ratio_window': ['qmin', 'delta_q'],
        **_EFFECTIVE_SPIN_HYPERS,
    },
    priors={
        'alpha': ([-4, 12], dist.Uniform),
        'beta': ([-2, 7], dist.Uniform),
        'mmin': ([2, 10], dist.Uniform),
        'mmax': ([30, 100], dist.Uniform),
        'delta_m': ([0, 10], dist.Uniform),
        'mpp': ([20, 50], dist.Uniform),
        'sigpp': ([1, 10], dist.Uniform),
        'lam': ([0, 1], dist.Uniform),
        'qmin': ([0.1], dist.Delta),
        'delta_q': ([0, 0.3], dist.Uniform),
        'mu_spin': ([0, 1], dist.Uniform),
        'var_spin': ([0.005, 0.25], dist.Uniform),
        'sig_tilt': ([0.1, 4], dist.Uniform),
        'zeta_tilt': ([0, 1], dist.Uniform),
        'lamb': ([-2, 10], dist.Uniform),
        'max_z': ([1.9], dist.Delta),
        'mu_x': ([-1, 1], dist.Uniform),
        'sig_x': ([0.005, 1.], dist.Uniform),
        'mu_xp': ([0, 1], dist.Uniform),
        'sig_xp': ([0.005, 1.], dist.Uniform),
    },
)


# ---------------------------------------------------------------------------
# GWTC-4 (O4a) -- the module-level registries in gwpop_models
# ---------------------------------------------------------------------------

gwtc4_default = CatalogDefaults(
    models=_gwpop.gwparameter_to_model,
    hyperparameters=_gwpop.gwparameter_to_hyperparameters,
    priors=_gwpop.default_priors,
    minima=_gwpop.bbh_minima,
    maxima=_gwpop.bbh_maxima,
)

# GWTC-5 reuses the GWTC-4 models. It is a distinct object rather than an alias
# so its priors can be adjusted without disturbing GWTC-4.
gwtc5_default = gwtc4_default.merge()


# ---------------------------------------------------------------------------
# GWTC-6 (O4c)
# ---------------------------------------------------------------------------

# Priors transcribed from gwtc6-population-models/priors/parametric_defaults.prior.
# lam_0/lam_1 there are DirichletElements of a 3-dimensional Dirichlet, which
# pixelpop passes as the single vector hyperparameter `lam_fractions`.
_GWTC6_PRIORS = {
    # mass
    'alpha_1': ([-4, 12], dist.Uniform),
    'alpha_2': ([-4, 12], dist.Uniform),
    'alpha_3': ([-4, 12], dist.Uniform),        # full spectrum only
    'beta': ([-2, 7], dist.Uniform),
    'mmax': ([300.], dist.Delta),
    'mlow_1': ([3, 10], dist.Uniform),
    'mlow_2': ([3, 10], dist.Uniform),
    'break_mass': ([20, 50], dist.Uniform),
    'break_mass_1': ([3, 20], dist.Uniform),    # full spectrum only
    'break_mass_2': ([20, 50], dist.Uniform),   # full spectrum only
    'mpp_1': ([5, 20], dist.Uniform),
    'mpp_2': ([25, 60], dist.Uniform),
    'sigpp_1': ([0, 10], dist.Uniform),
    'sigpp_2': ([0, 10], dist.Uniform),
    'delta_m_1': ([0, 10], dist.Uniform),
    'delta_m_2': ([0, 10], dist.Uniform),
    'lam_fractions': ([jnp.ones(3)], dist.Dirichlet),
    # GWTC-6 truncates the Gaussian peaks at 350, not pixelpop's default of 100.
    'gaussian_mass_maximum': ([350.], dist.Delta),
    # component spin magnitude
    'mu_1_chi': ([0, 0.25], dist.Uniform),
    'mu_2_chi': ([0.25, 1], dist.Uniform),
    'sigma_1_chi': ([0.005, 1], dist.Uniform),
    'sigma_2_chi': ([0.005, 1], dist.Uniform),
    'lamb_chi_1': ([0, 1], dist.Uniform),
    'lamb_chi_2': ([0, 1], dist.Uniform),
    'amax': ([1.], dist.Delta),
    # spin tilt (mu_spin/sigma_spin are the *tilt* mean and width here)
    'xi_spin': ([0, 1], dist.Uniform),
    'mu_spin': ([-1, 1], dist.Uniform),
    'sigma_spin': ([1e-1, 4e0], dist.Uniform),
    # redshift
    'lamb': ([-10, 10], dist.Uniform),
    'max_z': ([1.9], dist.Delta),
    # effective spins, for runs that pixelate chi_eff / chi_p
    'mu_x': ([-1, 1], dist.Uniform),
    'sig_x': ([0.005, 1.], dist.Uniform),
    'mu_xp': ([0, 1], dist.Uniform),
    'sig_xp': ([0.005, 1.], dist.Uniform),
}

_GWTC6_SHARED_MODELS = {
    # NOT PowerlawPlusPeak_MassRatio: same density, but that one normalizes on a
    # BBH-only fiducial grid and mis-normalizes by ~3.5x below m1 = 2. It stays the
    # GWTC-3/4/5 default so those remain reproducible.
    'mass_ratio': SmoothedPowerlaw_MassRatio,
    'redshift': PowerlawRedshiftPsi,
    't': tilt_model,
    **_EFFECTIVE_SPIN_MODELS,
    # GWTC-6 windows the mass ratio through the *secondary mass*, so that the
    # turn-on tracks m2 = m1 * q rather than q itself.
    'log_mass_1_window': PlanckWindow_PrimaryMass,
    'log_mass_2_window': PlanckWindow_SecondaryMass,
    'mass_ratio_window': PlanckWindow_SecondaryMass,
}

_GWTC6_SHARED_HYPERS = {
    # SmoothedPowerlaw_MassRatio(data, slope, minimum, delta_m)
    'mass_ratio': ['beta', 'mlow_2', 'delta_m_2'],
    'redshift': ['lamb', 'max_z'],
    't': ['mu_spin', 'sigma_spin', 'xi_spin'],   # tilt_model(data, mu, sig, zeta)
    'log_mass_1_window': ['mlow_1', 'delta_m_1'],
    'log_mass_2_window': ['mlow_2', 'delta_m_2'],
    'mass_ratio_window': ['mlow_2', 'delta_m_2'],
    **_EFFECTIVE_SPIN_HYPERS,
}

# two_gaussian_spin(data, mu_1_chi, mu_2_chi, sigma_1_chi, sigma_2_chi,
#                   lamb_chi_1, lamb_chi_2, amax)
_GWTC6_SPIN_HYPERS = ['mu_1_chi', 'mu_2_chi', 'sigma_1_chi', 'sigma_2_chi',
                      'lamb_chi_1', 'lamb_chi_2', 'amax']

# BrokenPowerlawPlusTwoPeaks_PrimaryMass(data, alpha_1, alpha_2, mmin, break_mass,
#     delta_m_1, lam_fractions, mpp_1, sigpp_1, mpp_2, sigpp_2, mmax,
#     gaussian_mass_maximum)
_GWTC6_MASS_HYPERS = ['alpha_1', 'alpha_2', 'mlow_1', 'break_mass', 'delta_m_1',
                      'lam_fractions', 'mpp_1', 'sigpp_1', 'mpp_2', 'sigpp_2',
                      'mmax', 'gaussian_mass_maximum']

gwtc6_default = CatalogDefaults(
    models={
        'mass_1': BrokenPowerlawPlusTwoPeaks_PrimaryMass,
        'log_mass_1': BrokenPowerlawPlusTwoPeaks_PrimaryMass,
        'a': two_gaussian_spin,
        **_GWTC6_SHARED_MODELS,
    },
    hyperparameters={
        'mass_1': list(_GWTC6_MASS_HYPERS),
        'log_mass_1': list(_GWTC6_MASS_HYPERS),
        'a': list(_GWTC6_SPIN_HYPERS),
        **_GWTC6_SHARED_HYPERS,
    },
    priors=_GWTC6_PRIORS,
)


# ---------------------------------------------------------------------------
# GWTC-6, full mass spectrum
# ---------------------------------------------------------------------------

# TripleBrokenPowerlawPlusTwoPeaks_PrimaryMass(data, alpha_1, alpha_2, alpha_3,
#     mmin, break_mass_1, break_mass_2, delta_m_1, lam_fractions, mpp_1, sigpp_1,
#     mpp_2, sigpp_2, mmax, gaussian_mass_maximum)
_GWTC6_FMS_MASS_HYPERS = ['alpha_1', 'alpha_2', 'alpha_3', 'mlow_1',
                          'break_mass_1', 'break_mass_2', 'delta_m_1',
                          'lam_fractions', 'mpp_1', 'sigpp_1', 'mpp_2',
                          'sigpp_2', 'mmax', 'gaussian_mass_maximum']

# The full spectrum has to reach the neutron stars. The BBH prior floor of
# 3 Msun puts *every* posterior sample of a BNS or NSBH (GW170817, GW190425,
# GW230529, GW200105, GW200115) at zero probability, and an event with no
# surviving samples makes its Monte-Carlo variance nan, so NUTS never finds a
# valid initial point. 1 Msun is the floor of the O4 injection set, so it is as
# low as the selection function can support. Capping mlow_1 at 3 also keeps
# mlow_1 < break_mass_1, which the mass model needs to stay finite.
FMS_MASS_MINIMUM = 1.
FMS_MASS_MAXIMUM = 3.

gwtc6_fms_default = CatalogDefaults(
    models={
        'mass_1': TripleBrokenPowerlawPlusTwoPeaks_PrimaryMass,
        'log_mass_1': TripleBrokenPowerlawPlusTwoPeaks_PrimaryMass,
        'a': two_gaussian_spin_fms,
        **_GWTC6_SHARED_MODELS,
    },
    hyperparameters={
        'mass_1': list(_GWTC6_FMS_MASS_HYPERS),
        'log_mass_1': list(_GWTC6_FMS_MASS_HYPERS),
        'a': list(_GWTC6_SPIN_HYPERS),
        **_GWTC6_SHARED_HYPERS,
    },
    priors={
        **_GWTC6_PRIORS,
        'mlow_1': ([FMS_MASS_MINIMUM, FMS_MASS_MAXIMUM], dist.Uniform),
        'mlow_2': ([FMS_MASS_MINIMUM, FMS_MASS_MAXIMUM], dist.Uniform),
    },
    minima={
        'mass_1': FMS_MASS_MINIMUM,
        'mass_2': FMS_MASS_MINIMUM,
        'log_mass_1': jnp.log(FMS_MASS_MINIMUM),
        'log_mass_2': jnp.log(FMS_MASS_MINIMUM),
        'log_mass_1_window': jnp.log(FMS_MASS_MINIMUM),
        'log_mass_2_window': jnp.log(FMS_MASS_MINIMUM),
    },
    maxima={
        'mass_1': 300.,
        'mass_2': 300.,
        'log_mass_1': jnp.log(300.),
        'log_mass_2': jnp.log(300.),
        'log_mass_1_window': jnp.log(300.),
        'log_mass_2_window': jnp.log(300.),
    },
)


GWTC_DEFAULTS = {
    'gwtc3': gwtc3_default,
    'gwtc4': gwtc4_default,
    'gwtc5': gwtc5_default,
    'gwtc6': gwtc6_default,
    'gwtc6_fms': gwtc6_fms_default,
}
