"""gta6-twitch-ccu-forecast-01: hourly Twitch CCU quantile forecast.

Reads every *.in case under $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out under $MAKEFUN_OUTPUT_DIR (else /output). Python 3.13, standard
library only, no network, deterministic.

Method (see README.md):
  1. Build each comparable launch's release-day profile as viewers by UTC
     hour 1..24, keep Tier: comparable donors with a positive hour-1 print.
  2. Weight donors by hour-1 scale (log-Gaussian), rolling-midnight unlock
     match, console-only platform match, and pre-launch peak vs hour-1.
  3. Per forecast hour, take log-ratios ln(v_h / v_1), shrink a weighted
     size slope, then read weighted empirical quantiles; scale the spread
     around the median and map back through hour 1's live print.
  4. Clip to (0, 1e9], enforce non-decreasing quantiles, round half-up to
     two decimals. Fall back to the house persistence baseline if a case
     has too few usable donors.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Grader runs with -P: put this file's directory back on sys.path first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))

# Kernel / calibration (leave-one-out on rolling-midnight comparables).
SIZE_BW = 1.0
ROLLING_BOOST = 1.2
CONSOLE_BOOST = 2.0
PRE_BW = 0.7
PRE_WEIGHT = True
SLOPE_SHRINK = 0.8
SLOPE_MIN = -0.5
SLOPE_MAX = 0.0
SPREAD_SCALE = 1.0
BASE_SIGMA = 0.15
MAX_VIEWERS = 1_000_000_000.0
MIN_Q = 0.01


def round2(x: float) -> float | int:
    """Half-up to two decimals; whole numbers as int for stable JSON."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def is_rolling(text: str) -> bool:
    low = (text or "").lower()
    return (
        ("midnight" in low and "local" in low)
        or "12 am local" in low
        or "rolling" in low
    )


def is_console_only(platforms: str) -> bool:
    return "pc" not in (platforms or "").lower()


def is_gta_history(comp: dict) -> bool:
    name = (comp.get("name") or "").lower()
    unlock = comp.get("unlock") or ""
    return "grand theft auto vi" in name or unlock.startswith("Not a launch")


def release_profile(comp: dict) -> dict[int, float]:
    """Map UTC release-day hours 1..24 to positive viewer counts."""
    try:
        start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    profile: dict[int, float] = {}
    for point in comp.get("hourly") or []:
        try:
            when = parse_time(point["from"])
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(viewers) or viewers <= 0:
            continue
        hour = int((when - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            profile[hour] = viewers
    return profile


def pre_peak_ratio(comp: dict, hour1: float) -> float | None:
    """Max viewers in the 24 h before day_start, divided by hour 1."""
    if hour1 <= 0:
        return None
    try:
        start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    peak = 0.0
    for point in comp.get("hourly") or []:
        try:
            when = parse_time(point["from"])
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if start - timedelta(hours=24) <= when < start and viewers > peak:
            peak = viewers
    if peak <= 0:
        return None
    return peak / hour1


def case_pre_peak_ratio(case: dict, hour1: float) -> float | None:
    """Live pre-launch peak from the case history relative to hour 1."""
    if hour1 <= 0:
        return None
    try:
        start = parse_time(case["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    peak = 0.0
    for point in case.get("history") or []:
        try:
            when = parse_time(point["at"])
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if start - timedelta(hours=24) <= when < start and viewers > peak:
            peak = viewers
    # history_hourly uses `from` instead of `at`
    for point in case.get("history_hourly") or []:
        try:
            when = parse_time(point["from"])
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if start - timedelta(hours=24) <= when < start and viewers > peak:
            peak = viewers
    if peak <= 0:
        return None
    return peak / hour1


def hour1_print(case: dict) -> float:
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            try:
                viewers = float(row["viewers"])
            except (KeyError, TypeError, ValueError):
                break
            if math.isfinite(viewers):
                return max(1.0, viewers)
    return 1.0


def weighted_quantile(samples: list[tuple[float, float]], level: float) -> float:
    """Weighted quantile via mid-point CDF of sorted (value, weight)."""
    ordered = sorted(samples)
    total = sum(weight for _, weight in ordered)
    if not ordered or total <= 0:
        return ordered[len(ordered) // 2][0] if ordered else 0.0

    centers: list[tuple[float, float]] = []
    cumulative = 0.0
    for value, weight in ordered:
        centers.append((cumulative + weight * 0.5, value))
        cumulative += weight

    target = level * total
    if target <= centers[0][0]:
        return centers[0][1]
    for (left_pos, left_value), (right_pos, right_value) in zip(centers, centers[1:]):
        if target <= right_pos:
            if right_pos == left_pos:
                return right_value
            frac = (target - left_pos) / (right_pos - left_pos)
            return left_value + frac * (right_value - left_value)
    return centers[-1][1]


def donor_weight(
    donor: dict,
    target_ln1: float,
    target_rolling: bool,
    target_console: bool,
    target_pre: float | None,
) -> float:
    dist = ((donor["ln1"] - target_ln1) / SIZE_BW) ** 2
    if PRE_WEIGHT and target_pre is not None and donor["pre"] is not None:
        t_log = math.log(max(1e-6, target_pre))
        d_log = math.log(max(1e-6, donor["pre"]))
        dist += ((d_log - t_log) / PRE_BW) ** 2
    weight = math.exp(max(-80.0, -0.5 * dist))
    if target_rolling and donor["rolling"]:
        weight *= ROLLING_BOOST
    if target_console and donor["console"]:
        weight *= CONSOLE_BOOST
    return weight


def build_donors(case: dict) -> list[dict]:
    donors: list[dict] = []
    for comp in case.get("comparables") or []:
        unlock = comp.get("unlock") or ""
        if "Tier: comparable." not in unlock:
            continue
        if is_gta_history(comp):
            continue
        profile = release_profile(comp)
        base = profile.get(1)
        if base is None or base <= 0:
            continue
        donors.append(
            {
                "profile": profile,
                "ln1": math.log(base),
                "base": base,
                "rolling": is_rolling(unlock),
                "console": is_console_only(comp.get("platforms") or ""),
                "pre": pre_peak_ratio(comp, base),
            }
        )
    return donors


def baseline_quantiles(p1: float, hour: int) -> list[float | int]:
    qs: list[float | int] = []
    floor = 0.0
    for z in Z:
        value = p1 * math.exp(z * BASE_SIGMA * math.sqrt(hour - 1))
        value = min(MAX_VIEWERS, max(MIN_Q, round2(value)))
        if value < floor:
            value = floor
        floor = float(value)
        qs.append(value)
    return qs


def clean_quantiles(raw: list[float], p1: float) -> list[float | int]:
    """Clip, round half-up, and force non-decreasing positive quantiles."""
    cleaned: list[float | int] = []
    prev = 0.0
    for value in raw:
        if not math.isfinite(value):
            value = p1
        value = min(MAX_VIEWERS, max(MIN_Q, value))
        if value < prev:
            value = prev
        rounded = round2(value)
        if float(rounded) < prev:
            # Rounding can step below the previous quantile; stick to prev.
            rounded = int(prev) if float(prev).is_integer() else prev
        prev = float(rounded)
        cleaned.append(rounded)
    return cleaned


def hour_log_quantiles(
    donors: list[dict],
    hour: int,
    target_ln1: float,
    target_rolling: bool,
    target_console: bool,
    target_pre: float | None,
) -> list[float] | None:
    points: list[tuple[float, float, float]] = []
    for donor in donors:
        profile = donor["profile"]
        if hour not in profile:
            continue
        weight = donor_weight(
            donor, target_ln1, target_rolling, target_console, target_pre
        )
        if weight <= 0:
            continue
        log_ratio = math.log(profile[hour] / donor["base"])
        points.append((donor["ln1"], log_ratio, weight))
    if len(points) < 3:
        return None

    total_w = sum(w for _, _, w in points)
    mean_x = sum(w * x for x, _, w in points) / total_w
    mean_y = sum(w * y for _, y, w in points) / total_w
    var_x = sum(w * (x - mean_x) ** 2 for x, _, w in points)
    if var_x > 0:
        raw_slope = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in points) / var_x
    else:
        raw_slope = 0.0
    slope = min(SLOPE_MAX, max(SLOPE_MIN, raw_slope * SLOPE_SHRINK))

    adjusted = [(y - slope * (x - target_ln1), w) for x, y, w in points]
    center = weighted_quantile(adjusted, 0.5)
    logs = [
        center + SPREAD_SCALE * (weighted_quantile(adjusted, level) - center)
        for level in LEVELS
    ]
    for i in range(1, len(logs)):
        if logs[i] < logs[i - 1]:
            logs[i] = logs[i - 1]
    return logs


def forecast_case(case: dict) -> dict:
    p1 = hour1_print(case)
    target_ln1 = math.log(p1)
    target_rolling = is_rolling(case.get("notes") or "")
    # Practice and final targets are console launches (GTA VI is console-only).
    target_console = True
    target_pre = case_pre_peak_ratio(case, p1)
    donors = build_donors(case)

    forecast_hours = case.get("forecast") or [{"hour": h} for h in HOURS]
    rows: list[dict] = []
    for row in forecast_hours:
        try:
            hour = int(row["hour"])
        except (KeyError, TypeError, ValueError):
            continue
        if hour not in HOURS:
            continue
        logs = hour_log_quantiles(
            donors, hour, target_ln1, target_rolling, target_console, target_pre
        )
        if logs is None:
            qs = baseline_quantiles(p1, hour)
        else:
            raw = []
            for log_ratio in logs:
                # Cap exponent so p1 * exp(log_ratio) stays in (0, 1e9].
                capped = min(math.log(MAX_VIEWERS / p1), max(-80.0, log_ratio))
                raw.append(p1 * math.exp(capped))
            qs = clean_quantiles(raw, p1)
        rows.append({"hour": hour, "q": qs})

    rows.sort(key=lambda r: r["hour"])
    if len(rows) != 23:
        # Guarantee a valid shape even on malformed forecast lists.
        rows = [{"hour": h, "q": baseline_quantiles(p1, h)} for h in HOURS]
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(case, dict):
                case = {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            case = {}
        result = forecast_case(case)
        payload = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
