# Migration chronicle: per-event variable PE sample counts (ragged posteriors)

## Why

PixelPop historically loaded events with `gwpopulation_pipe.data_collection.load_all_events`,
which **downsamples every event to the smallest event's sample count**
(`data_collection.py:616-625`: `n_samples = min(len(post) ...)` then `DataFrame.sample`).
That discards samples from well-measured events.

This branch (`ragged-pe-samples`) lets each event keep its natural number of PE samples.
The representation change is:

| | before | after |
|---|---|---|
| public `PixelPopData.posteriors` | `dict[str, (Nobs, Nsample)]` | **`list[dict[str, (Nsample_i,)]]`** (one dict per event) |
| internal event handling | one rectangular array, `logsumexp(..., axis=1)` | **loop over the list of event dicts**, per-event `logsumexp` |
| injections | single rectangular set | **unchanged** (single rectangular set) |

## Done in this branch (Part A)

- **`pixelpop/utils/collection.py`** (new):
  - `load_all_events_no_downsample(args, ...)` — copy of upstream `load_all_events` with
    the downsampling block removed; returns `dict[event_name -> DataFrame]` of full length.
  - `posteriors_to_pytree(posteriors, parameters)` — converts that dict to
    `(event_dicts, event_names)`, the list-of-dicts pytree.
- **`pixelpop/utils/data.py`**:
  - `PixelPopData.posteriors` typed/documented as a list of per-event dicts.
  - New helpers `build_bin_axes`, `bin_dataset`, `flag_out_of_range`, `flag_injection_free`
    (the old `place_in_bins`/`check_bins` are kept intact for backward compatibility).
  - `preprocess_cosmology` and `__post_init__` loop over events; `event_bins`
    (and IID `event_bins_1`/`event_bins_2`) are now **lists** of per-event bin tuples;
    `Nobs = len(posteriors)`.
- **`pixelpop/utils/__init__.py`**: guarded re-export of `collection` (optional dep).

## Part B — implemented

> Status: **implemented.** Each entry names the file and the change made. Two caveats
> (B4 validation, and the experimental model) are documented at the end.

### B1. Likelihood — `pixelpop/models/gwpop_models.py` ✅
- Added `_per_event_moments(event_weights)` which accepts either a ragged `list`/`tuple`
  of 1-D per-event weight arrays or a rectangular `(n_events, n_samples)` array, and
  returns `(n_events, counts, numerators, square_sums)`. `rate_likelihood` now uses it
  (per-event `counts` replaces the scalar `minimum_length`). The **VT / injection** branch
  (`denominator_weights`, `square_sum`, `vt_neff`, `nexp`, `vt_ln_likelihood*`) is
  unchanged. Both list and 2-D inputs are supported (back-compatible).
- `hierarchical_likelihood` (the jitted, currently-unused function previously called
  "ess_rate_likelihood" in the plan) was left rectangular-only — nothing calls it.

### B2. Model assembly — `pixelpop/models/probabilistic.py` ✅
- Split sampling from per-dataset weight accumulation:
  - `sample_pixelpop_field()` performs all ICAR `sample`/`factor`/`deterministic` calls
    once (incl. the `log_rate` normalization and `log_marginal_*`); returns
    `{'skip', 'mrd', 'normalization'}` (or `{'skip': True, 'R': ...}` when skipped).
  - `pixelpop_log_weight(bins, field)` returns the additive pixelized log-weight for one
    dataset's bins (IID-aware).
  - `sample_parametric_hyperparameters()` draws the hyperparameters + constraint factors
    once; `parametric_log_weight(data, sample)` evaluates the parametric contribution for
    one dataset.
- `probabilistic_model` samples the shared field/hyperparameters once, evaluates the
  injection weights once (rectangular), then **loops over the list of event dicts** to
  build a per-event 1-D weight array, and passes the list to `rate_likelihood`. All
  NumPyro site names are preserved (`merger_rate_density`, `log_rate`, `lnsigma`,
  `log_marginal_*`, parametric hyperparameters, deterministics).
- `get_initial_value`, `get_table_size`, `inference_loop`: unchanged — the list pytree
  flows through `model_kwargs` as a pytree arg. Confirmed by tracing + a short NUTS run.

### B3. Post-processing — `pixelpop/result/post_processing.py` ✅
- `PixelPopRateFunction`: stores the per-event bin **list** for `dataset_type='posteriors'`
  and rectangular bins for `'injections'`. `__call__(dataset, hyperparameters, bins=None)`
  now takes explicit `bins` (the caller supplies per-event bins; `None` falls back to the
  stored injection bins). `log_prob_parametric_model`/`log_rate_pixelpop` take shape/bins
  explicitly.
- `resample_posteriors`: loops over events, reweighting each event's ragged sample array
  independently. Returns `reweight_iloc` shaped `(nsamples, Nobs)` (column `ii` indexes
  event `ii`'s own samples) and per-event `neffs` shaped `(Nobs,)`.
- `reweight_events_and_injections`: indexing updated to
  `posteriors[ii][gw][event_iloc[:,ii]]` and `posteriors[0].keys()`.

### B4. Validation — `pixelpop/result/validate.py` ✅ (with caveat)
- Stacks the ragged event list into a rectangular dict for the external
  `population_error.error_statistics` API, and stacks the per-event bins onto the rate
  function. **Caveat:** `population_error.error_statistics` reads
  `event_posteriors['prior'].shape[0]` and calls `model(dataset, params)` without per-event
  bins, so it only supports **equal per-event sample counts**. `compute_error_statistics`
  now raises a clear `NotImplementedError` for genuinely ragged counts; full ragged support
  would require changing the external `population_error` package.

### B5. Save — `pixelpop/result/save_popsummary.py` ✅ (no change needed)
- Derives event names from data-dir globbing and delegates resampling to
  `reweight_events_and_injections` (B3). It does not index `pixelpop_data.posteriors`
  arrays directly, so no change was required.

### B6. Examples — `examples/mass1_redshift.py`, `examples/masses.py` ✅
- Now build a **list of per-event dicts** and apply `convert_*` / `clean_par` per event
  (the helpers are unchanged; only the call sites). A comment points at
  `pixelpop.utils.collection.posteriors_to_pytree` as an alternative.

### B7. Tests — `test/utils/data_test.py` ✅
- Added `TestRaggedPixelPopData.test_ragged_construction`: builds a `PixelPopData` with
  unequal per-event counts (50 / 120 / 300) and asserts `Nobs`, `len(event_bins)`, and
  per-event `ln_dVTc` / `log_prior` / bin shapes. Suite: **8 passed**.
- End-to-end (CPU env, not a committed test): a short NUTS run on ragged data yields a
  finite `log_likelihood`, and `resample_posteriors` returns `(nsamples, Nobs)` /
  `(Nobs,)` shapes.

## Notes / gotchas
- `place_in_bins` and `check_bins` are retained but no longer used by `PixelPopData`;
  `models/probabilistic.py:15` and `experimental/probabilistic.py:8` import `place_in_bins`
  but never call it (pre-existing dead imports).
- `jnp.ravel_multi_index(..., mode='clip')` in `flag_injection_free` keeps already-flagged
  out-of-range indices from raising; their prior is `inf` regardless.
- Injections are intentionally **not** ragged — there is a single injection set shared by
  all events, so all VT-side code stays rectangular.
- **`pixelpop/experimental/probabilistic.py` migrated** too: `prior_probabilistic_model`
  now uses the same split (`sample_pixelpop_field` / `pixelpop_log_weight` /
  `sample_parametric_hyperparameters` / `parametric_log_weight`) and loops over the event
  list. The SVI / NeuTra inference helpers were untouched — they pass the `posteriors` list
  through `model_kwargs` as a pytree. Verified by tracing with a finite log-likelihood.
