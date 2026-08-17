"""Ordered hyperparameter pairs, reparameterized onto the triangle they live on."""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


#: Ordered hyperparameter pairs, ``{site stem: (upper, lower)}``. The stem names
#: the site the pair is sampled through, ``stem + ORDERED_PAIR_SUFFIX``.
ORDERED_HYPERPARAMETER_PAIRS = {'mlow': ('mlow_1', 'mlow_2')}

#: Suffix of the unit-square site an ordered pair is sampled through.
ORDERED_PAIR_SUFFIX = '_frac'

#: ``{(upper, lower): site}``, the reverse of the mapping above.
_PAIR_SITES = {members: stem + ORDERED_PAIR_SUFFIX
               for stem, members in ORDERED_HYPERPARAMETER_PAIRS.items()}


def ordered_pair_site(pair):
    """
    Name of the length-2 site an ordered pair is sampled through.

    Parameters
    ----------
    pair : tuple of str
        ``(upper, lower)``, as the keys of :func:`ordered_pair_bounds`.

    Returns
    -------
    site : str
    """
    try:
        return _PAIR_SITES[tuple(pair)]
    except KeyError:
        raise KeyError(
            f"{tuple(pair)} is not an ordered hyperparameter pair; known pairs "
            f"are {sorted(_PAIR_SITES)}"
        ) from None


def ordered_pair_bounds(priors):
    """
    Bounds of the ordered hyperparameter pairs a run samples.

    A pair qualifies only if both members have a Uniform prior.

    Parameters
    ----------
    priors : dict
        ``{name: (args, distribution)}``, as in ``PixelPopData.priors``.

    Returns
    -------
    bounds : dict
        ``{(upper, lower): (lo, hi)}`` for every qualifying pair in
        :data:`ORDERED_HYPERPARAMETER_PAIRS`.

    Raises
    ------
    ValueError
        If the two members of a pair have different Uniform ranges.
    """
    bounds = {}
    for upper, lower in ORDERED_HYPERPARAMETER_PAIRS.values():
        if not {upper, lower} <= set(priors):
            continue
        if any(priors[name][1].__name__ != 'Uniform' for name in (upper, lower)):
            continue
        ranges = {name: tuple(float(v) for v in priors[name][0])
                  for name in (upper, lower)}
        if ranges[upper] != ranges[lower]:
            raise ValueError(
                f"{upper} and {lower} are an ordered pair and must share a prior "
                f"range, but got {ranges[upper]} and {ranges[lower]}"
            )
        bounds[(upper, lower)] = ranges[lower]
    return bounds


def reparameterized_sites(priors):
    """
    Map each ordered hyperparameter onto the site NUTS actually samples.

    Parameters
    ----------
    priors : dict
        ``{name: (args, distribution)}``, as in ``PixelPopData.priors``.

    Returns
    -------
    sites : dict
        ``{hyperparameter: sampled site name}``, empty if this run
        reparameterizes nothing. Both members of a pair map to the same site.
    """
    return {name: ordered_pair_site(pair)
            for pair in ordered_pair_bounds(priors) for name in pair}


def sample_ordered_pair(upper, lower, lo, hi):
    """
    Sample a pair uniformly on the triangle ``{lo <= lower <= upper <= hi}``.

    Parameters
    ----------
    upper, lower : str
        Hyperparameter names, ``lower <= upper``. Each is emitted as a
        ``numpyro.deterministic`` site of the same name.
    lo, hi : float
        Shared bounds of the pair.

    Returns
    -------
    upper_value, lower_value : jnp.ndarray
        The sampled pair, in the same order as the names.
    """
    # Beta(1, 2) on `lower`'s fraction cancels the 1/(hi - lower) that drawing
    # `upper` above it contributes, making the joint flat on the triangle.
    # `upper`'s fraction is flat, i.e. Beta(1, 1), so both fit in one site:
    # index 0 is `lower`'s, index 1 is `upper`'s.
    frac = numpyro.sample(ordered_pair_site((upper, lower)),
                          dist.Beta(jnp.array([1., 1.]), jnp.array([2., 1.])))
    lower_value = lo + (hi - lo) * frac[0]
    upper_value = lower_value + (hi - lower_value) * frac[1]
    return (numpyro.deterministic(upper, upper_value),
            numpyro.deterministic(lower, lower_value))


def ordered_pair_initial_value(priors, plausible_hyperparameters=None, offset=0.01):
    """
    Initial value for the unit-square site of each ordered pair.

    Defaults to just inside the low corner of the triangle, ``lower = lo +
    offset`` and ``upper = lo + 2 * offset``. The exact corner is unusable: both
    fractions are ``0`` there, which is ``-inf`` unconstrained.

    Parameters
    ----------
    priors : dict
        ``{name: (args, distribution)}``, as in ``PixelPopData.priors``.
    plausible_hyperparameters : dict, optional
        Overrides the default corner for any pair whose members are both present.
        Values outside the triangle are pulled back inside it.
    offset : float, optional
        Distance from the corner, in the units of the hyperparameters themselves
        (default 0.01).

    Returns
    -------
    initial_value : dict
        ``{site: array of the two fractions}``, ready to merge into the dict
        handed to ``numpyro.infer.init_to_value``. Values are arrays, not scalars.
    """
    plausible_hyperparameters = plausible_hyperparameters or {}
    initial_value = {}
    for (upper, lower), (lo, hi) in ordered_pair_bounds(priors).items():
        lower_value, upper_value = lo + offset, lo + 2 * offset
        if {upper, lower} <= set(plausible_hyperparameters):
            lower_value = float(plausible_hyperparameters[lower])
            upper_value = float(plausible_hyperparameters[upper])
        # keep both fractions strictly inside (0, 1), even for a pair that came
        # in unordered or out of range
        lower_value = min(max(lower_value, lo + offset), hi - 2 * offset)
        upper_value = min(max(upper_value, lower_value + offset), hi - offset)
        initial_value[ordered_pair_site((upper, lower))] = jnp.array([
            (lower_value - lo) / (hi - lo),
            (upper_value - lower_value) / (hi - lower_value),
            ])
    return initial_value
