# Structural break, real-time: final entry #141 ("MLPMIR")

Our final submission to the ADIA Lab Structural Break Challenge: Real Time
Edition, run on CrunchDAO. Team **regime-change**, Windyview AI Lab.
Authors: Katlun Yong and Ho Yin Yong.

This is the October 2026 final entry, exactly as it was submitted. Later
post-competition work is not part of this repository.

## The competition

Each example is a univariate time series with a known "historical" stretch
followed by an "online" stretch that arrives one step at a time. The task is to
output, at every step, a score for how likely it is that the series has had a
structural break (a change in its data-generating process) after the historical
part. The score is a ranking metric, and the platform runs your code in real
time on its own servers, so the code has to be fast and must never crash.

## What this entry does

Everything is in `submission/main.py` (plus helper modules next to it). At each
step it does roughly this:

1. **Feature streams.** The online series is fed through a set of streaming
   statistics that update cheaply as each new point arrives: a rich set of
   distribution and change features on the raw series, the same features on an
   AR-whitened version of the series, calibrated innovation statistics,
   volatility (EWMA, GARCH-style) and variance-shift CUSUM statistics,
   extreme-value and trimmed-mean gaps, variance-ratio features and a few more
   (`consol_stream.py`, `rich_features_v4.py`, `vol_stream.py`,
   `garch_stream.py`, `glr_stream.py`, `ev_stream.py`, `trimgap_stream.py`,
   `vr_stream.py`, `compound_streams.py`, `gene_stream.py`, and others).
2. **LightGBM.** A gradient-boosted tree model scores that feature vector.
3. **MLP ensemble ("MLP mirror").** 14 small neural nets score a related
   feature view and are blended with the trees at a low weight. This is the
   change that defines build #141: the nets are retrained on the real series
   plus sign-flipped (mirrored) copies of them, so the score does not depend on
   which direction the shift goes. The trees and the other parts are unchanged
   from the previous build.
4. **Sign-flip averaging at test time.** The series is also scored with its
   sign flipped, and the two raw scores are averaged before the final
   mapping to a rank-like score.
5. **Sequence legs.** TCN / GRU / variance-ratio / evolved legs are in the code
   (`seqleg.py`, `tcn_build.py`, `gru_build.py`, `genes.py`) but several ship
   at weight 0, so they do not change the output. The Chronos and robust-stream
   branches are likewise switched off (weight 0) in this build.

`main.py` contains both `train()` and `infer()`. Many comments in it are our
working notes from the competition (what was tried, what failed on the board),
left as written.

## Results

- Selected as our final entry.
- Public test set: **64.624**
- Final out-of-sample: **64.616**, ranked **67th of 798**.

Technical note: https://windyviews.com/sbrt/

We presented a poster on this work at the ADIA Lab Symposium, Abu Dhabi,
October 2026.

## How to run

You need the organiser's tooling and your own copy of the competition data
(the dataset is not redistributed here; get it from CrunchDAO after accepting
their terms).

```
pip install crunch-cli
crunch setup --help          # follow the organiser's setup for this challenge
cd submission
crunch test                  # local train + infer check against the data you downloaded
```

Python dependencies are in `submission/requirements.txt` (LightGBM is pinned to
4.6.0). The shipped model is `submission/resources/model.joblib`; it is loaded
with `joblib` (pickle), so only load it from a source you trust. We have not
re-run `crunch test` while preparing this repository; the files are the ones
that were submitted.

## Layout

```
submission/               the package exactly as submitted
  main.py                 train() and infer()
  *_stream.py, ...        feature streams and legs
  requirements.txt
  resources/model.joblib  the shipped model (14.9 MB)
build/                    scripts used to rebuild the model for this entry
  mlpmir_build.py         retrains the MLP nets on real + mirrored rows
  patch_main_mlpmir.py    the edit that turned the previous build's main.py into this one
  aa_gate.py, preflight_mirror_rows.py   checks run before pushing
```

The `build/` scripts have hard-coded paths from our own machines (for example
`/data/mlpmir`, `/root/ml-comp`) and data locations you would need to change.
`patch_main_mlpmir.py` expects the previous build's `main.py`, which is not
included; the patched result is `submission/main.py`.

## Credits

Katlun Yong and Ho Yin Yong, team regime-change, Windyview AI Lab.

## License

MIT, see `LICENSE`.
