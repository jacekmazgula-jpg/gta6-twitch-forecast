# GTA VI Twitch CCU forecast

Entry for [make.fun bounty `gta6-twitch-ccu-forecast-01`](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): after the first UTC hour of Grand Theft Auto VI’s release day, forecast Twitch category viewers for hours 2–24 as five quantiles (`q05`, `q25`, `q50`, `q75`, `q95`).

## Run

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Python 3.13, standard library only. No network. Two runs on the same case write identical bytes.

## Method

Console launches that unlock at local midnight roll around the planet. The UTC release day therefore has a repeatable shape once each curve is scaled by its own first-hour audience: a modest hour 1 (only early time zones live), a climb as Europe and the Americas unlock, then an evening peak.

`main.py` transfers that shape from the case’s `comparables`:

1. **Donors.** Keep launches marked `Tier: comparable.` with a positive hour-1 print. Drop the GTA VI category’s own pre-release history (it is not a launch day).

2. **Weights.** Each donor’s weight is a product of:
   - log hour-1 scale similarity (Gaussian kernel, bandwidth 1.0),
   - a boost when both target and donor are rolling-midnight unlocks,
   - a boost when both look console-only (no PC in `platforms`),
   - similarity of the pre-launch 24 h peak relative to hour 1 (from the case `history` for the target, from each comparable’s hourly series for donors).

3. **Centre and spread.** For each hour `h` in 2..24, take log-ratios `ln(v_h / v_1)`. Fit a weighted size slope of those ratios on log hour-1, shrink it (0.8) and clamp it to `[-0.5, 0]`, then adjust every donor onto the target’s hour-1 scale. The five quantiles are weighted empirical quantiles of the adjusted log-ratios (spread scale 1.0 around the median).

4. **Map back.** Multiply `exp(quantile)` by the live hour-1 print, clip to `(0, 1e9]`, round half-up to two decimals, and force non-decreasing levels. If an hour has fewer than three usable donors, fall back to the house persistence baseline (`p1 * exp(z * 0.15 * sqrt(h - 1))`).

The pinball loss on the log scale rewards an honest spread as well as an accurate median, so the forecast keeps cross-launch dispersion instead of collapsing to a point estimate.

## Layout

| Path | Role |
| --- | --- |
| `main.py` | Forecaster entrypoint |
| `.makefun/owner.json` | Wallet claim for make.fun |
| `LICENSE` | MIT |
