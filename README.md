# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jacekmazgula-jpg/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jacekmazgula-jpg/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Offline quantile forecaster for GTA VI’s launch-day Twitch viewers: it reads hour 1’s print, transfers release-day shapes from comparable launches (weighted toward rolling-midnight console unlocks and adjusted with live pre-launch signals), and writes five quantiles an hour for hours 2–24.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=zvsv8k75&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

## Method

Console launches that unlock at local midnight roll around the planet. The UTC release day therefore has a repeatable shape once each curve is scaled by its own first-hour audience: a modest hour 1 (only early time zones live), a climb as Europe and the Americas unlock, then an evening peak.

`main.py` transfers that shape from the case’s `comparables`:

1. **Donors.** Keep launches marked `Tier: comparable.` with a positive hour-1 print. Drop the GTA VI category’s own pre-release history (it is not a launch day). Prefer rolling-midnight console launches and keep at least eight active donors so a huge hour-1 print cannot collapse the kernel onto one or two PC worldwide unlocks.

2. **Weights.** Each donor’s weight is a product of:
   - log hour-1 scale similarity (Gaussian kernel; bandwidth widens when the target sits above every donor),
   - a boost when both target and donor are rolling-midnight unlocks,
   - a boost when both look console-only (no PC in `platforms`),
   - similarity of the pre-launch 24 h peak relative to hour 1 (from the case `history` for the target, from each comparable’s hourly series for donors).

3. **Centre and spread.** For each hour `h` in 2..24, take log-ratios `ln(v_h / v_1)`. Fit a weighted size slope of those ratios on log hour-1, shrink it and clamp it, then adjust every donor onto the target’s hour-1 scale. The five quantiles are weighted empirical quantiles of the adjusted log-ratios. A minimum log half-width then grows through the day (~0.15 at hour 2, ≥0.3 from hour 6, ~0.45–0.6 for hours 15–24) so late-day bands stay honest when hour 1 is far above the donor pool.

4. **Map back.** Multiply `exp(quantile)` by the live hour-1 print, clip to `(0, 1e9]`, round half-up to two decimals, and force non-decreasing levels. If an hour has fewer than three usable donors, fall back to the house persistence baseline (`p1 * exp(z * 0.15 * sqrt(h - 1))`).

The pinball loss on the log scale rewards an honest spread as well as an accurate median, so the forecast keeps cross-launch dispersion instead of collapsing to a point estimate.

## Layout

| Path | Role |
| --- | --- |
| `main.py` | Forecaster entrypoint |
| `.makefun/owner.json` | Wallet claim for make.fun |
| `LICENSE` | MIT |
