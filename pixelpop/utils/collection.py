"""
Posterior collection for PixelPop with per-event variable sample counts.

This module wraps :mod:`gwpopulation_pipe.data_collection` so that posteriors can
be loaded *without* downsampling every event to the smallest event's sample count.
The upstream :func:`gwpopulation_pipe.data_collection.load_all_events` performs that
downsampling (it computes ``n_samples = min(len(post) for post in posteriors)`` and
calls ``DataFrame.sample(n_samples)`` on every event). We reproduce that function
verbatim except for the downsampling block, which is removed so each event keeps its
natural number of samples.

The loaded frames are then stacked into a single rectangular ``(Nobs, NPE)`` dict via
:func:`pixelpop.utils.data.posteriors_to_rectangular`: events with more than ``NPE``
samples are downsampled, events with fewer are padded with repeated samples whose
``prior`` is set to ``+inf`` (so the padded samples carry zero PixelPop weight). The
real per-event sample counts are returned alongside, for the variance / Neff
calculations. (:func:`posteriors_to_pytree` -- a list-of-per-event-dicts pytree -- is
also kept for callers that want the ragged representation.)
"""

import os
import json
import re
import pickle as pkl

import pandas as pd
from jax import numpy as jnp

# Reuse the upstream helpers unchanged; only the downsampling function is copied
# and modified below.
from gwpopulation_pipe.data_collection import (
    DEFAULT_PARAMETER_MAPPING,
    _load_batch_of_meta_files,
    evaluate_prior,
    logger,
    create_parser,
    set_backend,
    simulate_posteriors,
    plot_summary,
    dump_injection_data,
    resolve_arguments
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
    # only downsample events where Npe > args.samples_per_posterior
    posteriors = {
        post: pd.DataFrame(posteriors[post]).sample(
            min(
                len(pd.DataFrame(posteriors[post])),
                args.samples_per_posterior
                ), random_state=args.collection_seed
        ).reset_index(drop=True)
        for post in posteriors
        }
    # Evaluate the prior on shape-stable inputs. ``evaluate_prior`` jits the
    # (expensive) chi_eff/chi_p spin prior under the jax backend, and jax
    # recompiles every time the input length changes. Since we deliberately keep
    # each event at its natural (ragged) sample count, passing the frames as-is
    # would trigger a fresh, costly recompile for *every* event. Instead we pad
    # all events up to a common width (so the spin prior compiles exactly once),
    # evaluate, then trim each event's prior back to its real samples. The padded
    # rows are appended at the end, so the leading rows are untouched.
    real_lengths = {name: len(posteriors[name]) for name in posteriors}
    common_width = max(real_lengths.values())
    padded = {}
    for name, frame in posteriors.items():
        n = real_lengths[name]
        if n < common_width:
            extra = frame.sample(
                common_width - n, replace=True, random_state=args.collection_seed
            )
            padded[name] = pd.concat([frame, extra], ignore_index=True)
        else:
            padded[name] = frame
    padded = evaluate_prior(padded, args=args, dataset=label, meta=meta)
    posteriors = {
        name: padded[name].iloc[: real_lengths[name]].reset_index(drop=True)
        for name in padded
    }
    for key in args.parameters:
        for name in posteriors:
            if key not in posteriors[name]:
                raise KeyError(f"{key} not found for {name}")
    posteriors = {
        name: posteriors[name][args.parameters + ["prior"]] for name in posteriors
    }
    logger.info(f"Loaded {len(posteriors)} posteriors.")
    return posteriors

def gather_posteriors(args, save_meta_data=True):
    """
    Load in posteriors from files according to the command-line arguments.

    Parameters
    ----------
    args: argparse.Namespace
        Command-line arguments
    save_meta_data: bool
        Whether to write meta data about the loaded results to plain-text files.

    Returns
    -------
    posts: list
        List of `pd.DataFrame` posteriors.
    events: list
        Event labels
    """
    posteriors = load_all_events_no_downsample(
        args, save_meta_data=save_meta_data, ignore=args.ignore
    )
    posts = {}
    events = list()
    filenames = list()
    for filename in posteriors.keys():
        event = re.findall(r"(GW\d{6}_\d{6}|S\d{6}[a-z]*|GW\d{6})", filename)[-1]
        if event in events:
            logger.warning(f"Duplicate event {event} found, ignoring {filename}.")
            continue
        posts[filename] = posteriors[filename]
        events.append(event)
        filenames.append(filename)
    if save_meta_data:
        logger.info(f"Outdir is {args.run_dir}")
        with open(
            f"{args.run_dir}/data/{args.data_label}_posterior_files.txt", "w"
        ) as ff:
            for event, filename in zip(events, filenames):
                ff.write(f"{event}: {filename}\n")

    return posts, events


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


def main():
    parser = create_parser()
    args = parser.parse_args()
    resolve_arguments(args)

    if args.backend.lower() == "cupy":
        logger.warning(
            "cupy backend is not supported for data collection. Falling back to numpy."
        )
        backend = "numpy"
    else:
        backend = args.backend

    set_backend(backend)

    os.makedirs(f"{args.run_dir}/data", exist_ok=True)
    if args.injection_file is not None or args.sample_from_prior:
        posts = simulate_posteriors(args=args)
        events = [str(ii) for ii in range(len(posts))]
        posts = {e: p for e, p in zip(posts, events)}
    else:
        posts, events = gather_posteriors(args=args)
    logger.info(f"Using {len(posts)} events, final event list is: {', '.join(events)}.")
    posterior_file = f"{args.data_label}.pkl"
    logger.info(f"Saving posteriors to {posterior_file}")
    filename = os.path.join(args.run_dir, "data", posterior_file)

    from .data import posteriors_to_rectangular
    rect, event_counts, event_names = posteriors_to_rectangular(
        posts, args.parameters, args.samples_per_posterior, seed=args.collection_seed
    )
    with open(filename, 'wb') as ff:
        pkl.dump(
            {'posteriors': rect, 'event_counts': event_counts, 'event_names': event_names},
            ff,
        )

    if args.vt_file is not None:
        dump_injection_data(args)
