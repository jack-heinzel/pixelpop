# Per-event variable PE sample counts via padding

## Why

PixelPop historically loaded events with `gwpopulation_pipe.data_collection.load_all_events`,
which **downsamples every event to the smallest event's sample count**
(`data_collection.py:616-625`). That discards samples from well-measured events.

We want each event to keep (up to a cap of) its natural number of PE samples, **without**
giving up the rectangular `(Nobs, NPE)` array that lets the whole likelihood vectorize over
events. (An earlier attempt made `posteriors` a *list of per-event dicts* and looped over
events in Python; that defeats JAX vectorization and is slow. It was reverted.)

## Approach: pad to rectangular, track real counts

`posteriors` stays a **rectangular dict** `dict[str, (Nobs, NPE)]`. Events with more than
`NPE` samples are downsampled to `NPE`; events with **fewer** are **padded** up to `NPE`
by repeating randomly drawn rows, with the padded rows' `prior` set to `+inf`. A padded
sample's PixelPop importance weight is `exp(model − log_prior) = exp(finite − inf) = 0`, so
it drops out of every Monte-Carlo sum (this reuses the exact `prior = inf` mechanism
already used for out-of-range / injection-free samples, so gradients stay finite).

The **only** thing that must change versus equal-count behaviour: the per-event Monte-Carlo
mean and its variance / effective-sample (Neff) estimates must divide by each event's
**real** sample count `c_i`, not the padded width `NPE`. Padding adds zero-weight samples;
counting them would falsely inflate the apparent precision.

| | representation |
|---|---|
| `PixelPopData.posteriors` | `dict[str, (Nobs, NPE)]` (rectangular, padded) |
| `PixelPopData.event_counts` | `(Nobs,)` real per-event counts (defaults to `NPE`) |
| event handling | one rectangular array, `logsumexp(..., axis=1)` (unchanged) |
| injections | single rectangular set (unchanged) |

### Why `event_counts` is an explicit field, not derived from the prior

`clean_par` and the binning in `__post_init__` also set `log_prior = inf` for out-of-range
/ injection-free **real** samples. Those are genuine draws (zero weight) that must remain
in `c_i` — the MC denominator is the number of samples *drawn*. So `c_i` cannot be
re-derived from `isfinite(log_prior)` after cleaning/binning; it is captured at **pad time**
(by `posteriors_to_rectangular`) and passed through.

## Changes

### Data
- **`pixelpop/utils/data.py`**
  - New `posteriors_to_rectangular(posteriors, parameters, n_samples, seed=None)` →
    `(rect_dict, event_counts, event_names)`. Truncates/pads each event to `n_samples`;
    padded `prior = +inf`; returns the real per-event counts.
  - `PixelPopData` gains an optional `event_counts` field. In `__post_init__` it defaults
    to `jnp.full(Nobs, NPE)` (captured **before** `place_in_bins` adds inf priors).
  - Everything else (`place_in_bins`, `convert_*`, `clean_par`, the rectangular binning in
    `__post_init__`) is the pre-existing rectangular code.
- **`pixelpop/utils/collection.py`** (optional, needs `gwpopulation_pipe`)
  - `load_all_events_no_downsample` downsamples each event to
    `min(len, args.samples_per_posterior)` (no global min-downsample).
  - `gather_posteriors` / `main` build a rectangular set via
    `data.posteriors_to_rectangular` and pickle
    `{'posteriors', 'event_counts', 'event_names'}`.
  - `posteriors_to_pytree` (list-of-dicts) is retained for callers wanting the ragged form.

### Likelihood — `pixelpop/models/gwpop_models.py`
- `_per_event_moments(event_weights, event_counts=None)`: rectangular branch divides by
  `event_counts` when given (a length-`Nobs` array broadcasting over `axis=1`), else by
  `NPE`. `rate_likelihood(..., event_counts=None)` forwards it; `single_event_neffs`
  becomes per-event-correct automatically. `hierarchical_likelihood` gained the same
  optional kwarg.

### Model — `pixelpop/models/probabilistic.py`, `pixelpop/experimental/probabilistic.py`
- Reverted to the rectangular accumulation (`nonparametric_model` / `parametric_model`
  building a single `(Nobs, NPE)` `event_weights`). The only edit is passing
  `event_counts=pixelpop_data.event_counts` to `rate_likelihood`.

### Post-processing — `pixelpop/result/post_processing.py`
- Reverted to rectangular, **no count threading needed**: reweighting / Neff there are
  self-normalizing over the `NPE` columns, and padded rows have weight 0 (never resampled,
  contribute 0 to `Σw` and `Σw²`).

### Validation — `pixelpop/result/validate.py`
- Reverted to the rectangular body and passes `event_counts=pixelpop_data.event_counts` to
  `population_error.error_statistics` (the `event_counts` kwarg already exists upstream).
  No `pad_ragged_posteriors` / bin-padding needed — the posteriors are already rectangular.

### Examples / tests
- `examples/masses.py`, `examples/mass1_redshift.py`: load per-event arrays, call
  `posteriors_to_rectangular` (padding up to the largest event so no samples are
  discarded), and pass `event_counts` to `PixelPopData`.
- `test/utils/data_test.py`: `TestPosteriorsToRectangular` (padding shapes / counts /
  `prior=inf` tail / truncation) and `TestPaddedPixelPopData` (rectangular bins, preserved
  `event_counts`, and `event_counts=None` defaulting to `NPE`).

## Verification (CPU, against the working tree)
`JAX_PLATFORMS=cpu PYTHONPATH=<repo-root> <gwjax311-python> ...`
- `pytest test/utils/data_test.py` → 11 passed.
- Likelihood equivalence: `rate_likelihood` with `event_counts=None` == explicit all-`NPE`
  counts; padding an event with `prior=inf` (weight `−inf`) rows + correct `event_counts`
  reproduces the un-padded `log_likelihood`, `single_event_neffs`, and variance exactly;
  genuinely ragged counts give finite, manually-verified per-event numerators.
- End-to-end on padded data (counts 40/95/150): model trace → finite `log_likelihood` /
  `log_likelihood_variance`; `compute_error_statistics` → finite error/precision/accuracy.
