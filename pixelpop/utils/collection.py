"""
Posterior collection for PixelPop with per-event variable sample counts.

This module wraps :mod:`gwpopulation_pipe.data_collection` so that posteriors can
be loaded *without* downsampling every event to the smallest event's sample count.
The upstream :func:`gwpopulation_pipe.data_collection.load_all_events` performs that
downsampling (it computes ``n_samples = min(len(post) for post in posteriors)`` and
calls ``DataFrame.sample(n_samples)`` on every event). We reproduce that function
verbatim except for the downsampling block, which is removed so each event keeps its
natural number of samples.

The result is converted to a PixelPop "pytree" -- a list of per-event dictionaries,
each mapping parameter name to that event's 1-D sample array -- via
:func:`posteriors_to_pytree`.
"""

import os
import json

import pandas as pd
from jax import numpy as jnp

# Reuse the upstream helpers unchanged; only the downsampling function is copied
# and modified below.
from gwpopulation_pipe.data_collection import (
    DEFAULT_PARAMETER_MAPPING,
    _load_batch_of_meta_files,
    evaluate_prior,
    logger,
)


def load_all_events_no_downsample(args, save_meta_data=True, ignore=None):
    """
    Load posteriors for some/all events, keeping each event's full sample count.

    This is a copy of :func:`gwpopulation_pipe.data_collection.load_all_events` with
    the downsampling step removed. Upstream, the loaded posteriors are reduced to
    ``min(len(post) for post in posteriors)`` samples per event via
    ``DataFrame.sample(...)``; here we keep every sample so different events may have
    different numbers of posterior samples.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace containing the needed arguments, these are:
        - `sample_regex`: A dictionary of regex strings to search for the posterior files.
        - `preferred_labels`: A list of preferred labels to search for in the posterior files.
        - `parameters`: A list of parameters to extract from the posteriors.
        - `mass_prior`: The mass prior used in initial sampling.
        - `distance_prior`: The distance prior used in initial sampling.
        - `spin_prior`: The spin prior used in initial sampling.
        - `max_redshift`: The maximum redshift allowed in the sample.
    save_meta_data: bool
        Whether to write meta data about the loaded results to plain-text files.
    ignore: list
        List of strings to ignore in the file names to filter unwanted events.

    Returns
    -------
    posteriors: dict
        Dictionary of `pd.DataFrame` posteriors. Unlike the upstream function, the
        frames are *not* downsampled, so they may have differing lengths.
    """
    posteriors = dict()
    meta_data = dict()
    parameter_mapping = DEFAULT_PARAMETER_MAPPING.copy()
    if args.custom_parameter_mapping is not None:
        parameter_mapping.update(args.custom_parameter_mapping)
    logger.info("Loading posteriors...")
    for label, regex in args.sample_regex.items():
        posts, meta = _load_batch_of_meta_files(
            regex=regex,
            label=label,
            labels=args.preferred_labels,
            keys=args.parameters,
            ignore=ignore,
            mapping=parameter_mapping,
        )
        posteriors.update(posts)
        meta_data.update(meta)
    if save_meta_data:
        with open(os.path.join(args.run_dir, "data", "event_data.json"), "w") as ff:
            json.dump(meta_data, ff)
    # --- downsampling removed: keep each event's full set of samples ---
    posteriors = {post: pd.DataFrame(posteriors[post]) for post in posteriors}
    posteriors = evaluate_prior(posteriors, args=args, dataset=label, meta=meta)
    for key in args.parameters:
        for name in posteriors:
            if key not in posteriors[name]:
                raise KeyError(f"{key} not found for {name}")
    posteriors = {
        name: posteriors[name][args.parameters + ["prior"]] for name in posteriors
    }
    logger.info(f"Loaded {len(posteriors)} posteriors.")
    return posteriors


def posteriors_to_pytree(posteriors, parameters):
    """
    Convert a dict of per-event posterior frames into a PixelPop pytree.

    Parameters
    ----------
    posteriors : dict
        Mapping of event name -> `pd.DataFrame` (or any dict-like) of posterior
        samples, e.g. the output of :func:`load_all_events_no_downsample`. Each frame
        must contain every name in ``parameters`` plus ``"prior"``.
    parameters : list of str
        Parameter names to extract for each event. ``"prior"`` is always included.

    Returns
    -------
    event_dicts : list of dict
        One dictionary per event, mapping each parameter (and ``"prior"``) to a 1-D
        `jax.numpy` array of that event's samples. Event order follows iteration over
        ``posteriors``.
    event_names : list of str
        The event names, in the same order as ``event_dicts``.
    """
    keys = list(parameters)
    if "prior" not in keys:
        keys = keys + ["prior"]

    event_names = list(posteriors)
    event_dicts = [
        {key: jnp.asarray(posteriors[name][key]) for key in keys}
        for name in event_names
    ]
    return event_dicts, event_names
