"""gta6-twitch-ccu-forecast-01: hourly Twitch CCU quantile forecast.

Reads every *.in case under $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out under $MAKEFUN_OUTPUT_DIR (else /output). Python 3.13, standard
library only, no network, deterministic.

Method (see README.md):
  1. Build each comparable launch's release-day profile as viewers by UTC
     hour 1..24, keep Tier: comparable donors with a positive hour-1 print.
  2. Prefer rolling-midnight console launches and keep at least eight active
     donors so a huge hour-1 print cannot collapse onto one or two PC
     worldwide unlocks; widen the size kernel when extrapolating upward.
  3. Per forecast hour, take log-ratios ln(v_h / v_1), shrink a weighted
     size slope, then read weighted empirical quantiles; enforce a minimum
     log half-width that grows through the day, then map back through
     hour 1's live print.
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

# Kernel / calibration.
SIZE_BW = 1.0
SIZE_BW_EXTRAP = 3.0  # when target hour-1 sits above every donor
ROLLING_BOOST = 3.0
CONSOLE_BOOST = 2.0
CLIMB_BOOST = 2.5  # donors whose release day peaks well above hour 1
PRE_BW = 0.7
PRE_WEIGHT = True
SLOPE_SHRINK = 0.4  # soft size slope; softer still under extrapolation
SLOPE_MIN = -0.25
SLOPE_MAX = 0.0
SPREAD_SCALE = 1.0
MIN_ACTIVE_DONORS = 8
WEIGHT_FLOOR_FRAC = 0.12  # each active donor keeps at least this share of max weight
CLIMB_PEAK_LO = 1.25  # peak/hour1 above this looks like a rolling unlock day
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


def size_bandwidth(target_ln1: float, donors: list[dict]) -> float:
    """Widen the size kernel when the target sits above every donor."""
    if not donors:
        return SIZE_BW
    max_ln = max(d["ln1"] for d in donors)
    if target_ln1 > max_ln + 0.25:
        # Blend toward the wider bandwidth as the gap grows.
        gap = target_ln1 - max_ln
        blend = min(1.0, gap / 2.0)
        return SIZE_BW + blend * (SIZE_BW_EXTRAP - SIZE_BW)
    return SIZE_BW


def min_log_halfwidth(hour: int) -> float:
    """Minimum |ln q_outer - ln q50| that grows through the release day.

    ~0.15 at hour 2, at least 0.3 from hour 6, ~0.45-0.6 for hours 15-24.
    """
    if hour <= 2:
        return 0.15
    if hour < 6:
        return 0.15 + (0.30 - 0.15) * (hour - 2) / 4.0
    if hour < 15:
        return 0.30 + (0.45 - 0.30) * (hour - 6) / 9.0
    return 0.45 + (0.60 - 0.45) * min(9.0, hour - 15) / 9.0


def is_climbing(donor: dict) -> bool:
    """Release day that rises after hour 1 (rolling-midnight shape family)."""
    return donor["peak_ratio"] >= CLIMB_PEAK_LO


def prefer_score(donor: dict, target_ln1: float, extrapolating: bool) -> tuple:
    """Rank key: rolling + climbing console first; size only as a tiebreak."""
    return (
        1 if donor["rolling"] else 0,
        1 if is_climbing(donor) else 0,
        1 if donor["console"] else 0,
        # When extrapolating, prefer classic climb shape over nearest size.
        donor["peak_ratio"] if extrapolating else 0.0,
        -abs(donor["ln1"] - target_ln1),
    )


def select_active_donors(donors: list[dict], target_ln1: float) -> list[dict]:
    """Keep >=8 donors, favouring rolling-midnight climbing console launches."""
    if not donors:
        return []
    max_ln = max(d["ln1"] for d in donors)
    extrapolating = target_ln1 > max_ln + 0.25
    ranked = sorted(
        donors,
        key=lambda d: prefer_score(d, target_ln1, extrapolating),
        reverse=True,
    )
    if len(ranked) <= MIN_ACTIVE_DONORS:
        return list(ranked)

    chosen: list[dict] = []
    chosen_ids: set[int] = set()

    def take(donor: dict) -> None:
        if id(donor) not in chosen_ids:
            chosen.append(donor)
            chosen_ids.add(id(donor))

    # Seed with rolling + climbing launches (the GTA VI shape family).
    for donor in ranked:
        if donor["rolling"] and is_climbing(donor):
            take(donor)
    # Ensure console-only rolling launches are present even if peak is mild.
    for donor in ranked:
        if donor["rolling"] and donor["console"]:
            take(donor)
    # Fill to at least MIN_ACTIVE_DONORS from the preference ranking.
    for donor in ranked:
        if len(chosen) >= MIN_ACTIVE_DONORS:
            break
        take(donor)
    # If still short, take anyone left.
    for donor in ranked:
        if len(chosen) >= MIN_ACTIVE_DONORS:
            break
        take(donor)
    return chosen


def donor_weight(
    donor: dict,
    target_ln1: float,
    target_rolling: bool,
    target_console: bool,
    target_pre: float | None,
    bandwidth: float,
    extrapolating: bool,
) -> float:
    dist = ((donor["ln1"] - target_ln1) / bandwidth) ** 2
    if PRE_WEIGHT and target_pre is not None and donor["pre"] is not None:
        t_log = math.log(max(1e-6, target_pre))
        d_log = math.log(max(1e-6, donor["pre"]))
        dist += ((d_log - t_log) / PRE_BW) ** 2
    # Under extrapolation, damp size distance so huge hour-1 cannot crown
    # the single nearest PC worldwide unlock.
    if extrapolating:
        dist *= 0.35
    weight = math.exp(max(-80.0, -0.5 * dist))
    if target_rolling and donor["rolling"]:
        weight *= ROLLING_BOOST
    if target_console and donor["console"]:
        weight *= CONSOLE_BOOST
    if target_rolling and is_climbing(donor):
        weight *= CLIMB_BOOST
    # Flat/declining worldwide unlocks are poor GTA VI analogues.
    if target_rolling and donor["peak_ratio"] < 1.1:
        weight *= 0.25
    return weight


def floored_weights(raw: list[float]) -> list[float]:
    """Raise tiny weights so the mass cannot sit on one or two donors."""
    if not raw:
        return raw
    peak = max(raw)
    if peak <= 0:
        return [1.0] * len(raw)
    floor = peak * WEIGHT_FLOOR_FRAC
    return [max(w, floor) for w in raw]


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
        peak_ratio = max(profile.values()) / base
        donors.append(
            {
                "name": comp.get("name") or "",
                "profile": profile,
                "ln1": math.log(base),
                "base": base,
                "rolling": is_rolling(unlock),
                "console": is_console_only(comp.get("platforms") or ""),
                "pre": pre_peak_ratio(comp, base),
                "peak_ratio": peak_ratio,
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
            rounded = int(prev) if float(prev).is_integer() else prev
        prev = float(rounded)
        cleaned.append(rounded)
    return cleaned


def enforce_min_spread(logs: list[float], hour: int) -> list[float]:
    """Widen log quantiles around the median when the band is too tight."""
    center = logs[2]
    need = min_log_halfwidth(hour)
    half_lo = center - logs[0]
    half_hi = logs[4] - center
    scale_lo = need / half_lo if half_lo > 1e-12 and half_lo < need else 1.0
    scale_hi = need / half_hi if half_hi > 1e-12 and half_hi < need else 1.0
    # If a side collapsed (non-positive width), rebuild from normal z * need/1.645.
    rebuilt: list[float] = []
    for i, level_log in enumerate(logs):
        offset = level_log - center
        if offset < 0:
            if half_lo <= 1e-12:
                offset = Z[i] / 1.645 * need
            else:
                offset *= scale_lo
        elif offset > 0:
            if half_hi <= 1e-12:
                offset = Z[i] / 1.645 * need
            else:
                offset *= scale_hi
        rebuilt.append(center + offset)
    for i in range(1, len(rebuilt)):
        if rebuilt[i] < rebuilt[i - 1]:
            rebuilt[i] = rebuilt[i - 1]
    return rebuilt


def hour_log_quantiles(
    donors: list[dict],
    hour: int,
    target_ln1: float,
    target_rolling: bool,
    target_console: bool,
    target_pre: float | None,
    bandwidth: float,
) -> list[float] | None:
    active = select_active_donors(donors, target_ln1)
    max_ln = max((d["ln1"] for d in donors), default=target_ln1)
    extrapolating = target_ln1 > max_ln + 0.25
    staged: list[tuple[dict, float, float]] = []  # donor, log_ratio, raw weight
    for donor in active:
        profile = donor["profile"]
        if hour not in profile:
            continue
        weight = donor_weight(
            donor,
            target_ln1,
            target_rolling,
            target_console,
            target_pre,
            bandwidth,
            extrapolating,
        )
        if weight <= 0:
            continue
        log_ratio = math.log(profile[hour] / donor["base"])
        staged.append((donor, log_ratio, weight))

    if len(staged) < 3:
        return None

    weights = floored_weights([w for _, _, w in staged])
    points = [
        (donor["ln1"], log_ratio, weight)
        for (donor, log_ratio, _), weight in zip(staged, weights)
    ]
    point_donors = [donor for donor, _, _ in staged]

    total_w = sum(w for _, _, w in points)
    mean_x = sum(w * x for x, _, w in points) / total_w
    mean_y = sum(w * y for _, y, w in points) / total_w
    var_x = sum(w * (x - mean_x) ** 2 for x, _, w in points)
    if var_x > 0:
        raw_slope = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in points) / var_x
    else:
        raw_slope = 0.0
    # When extrapolating far above the pool, shrink the size slope harder so
    # PC worldwide unlocks cannot drag the median to ~0.4x by midday.
    shrink = SLOPE_SHRINK
    if extrapolating:
        shrink *= 0.4
    slope = min(SLOPE_MAX, max(SLOPE_MIN, raw_slope * shrink))

    adjusted = [(y - slope * (x - target_ln1), w) for x, y, w in points]
    center = weighted_quantile(adjusted, 0.5)

    # Blend toward climbing rolling-midnight curves when extrapolating: that
    # is the country-by-country unlock shape GTA VI will follow (US East at
    # hour 6, US prime time in hours 20-24).
    shape_samples = [
        (lr - slope * (ln1 - target_ln1), w)
        for (ln1, lr, w), donor in zip(points, point_donors)
        if donor["rolling"] and is_climbing(donor)
    ]
    if not shape_samples:
        shape_samples = [
            (lr - slope * (ln1 - target_ln1), w)
            for (ln1, lr, w), donor in zip(points, point_donors)
            if donor["rolling"]
        ]
    if shape_samples and (extrapolating or target_rolling):
        shape_center = weighted_quantile(shape_samples, 0.5)
        if extrapolating:
            gap = target_ln1 - max_ln
            blend = min(0.85, 0.45 + 0.2 * gap)
        else:
            blend = 0.35
        center = (1.0 - blend) * center + blend * shape_center

    logs = [
        center + SPREAD_SCALE * (weighted_quantile(adjusted, level) - center)
        for level in LEVELS
    ]
    for i in range(1, len(logs)):
        if logs[i] < logs[i - 1]:
            logs[i] = logs[i - 1]
    return enforce_min_spread(logs, hour)


def forecast_case(case: dict) -> dict:
    p1 = hour1_print(case)
    target_ln1 = math.log(p1)
    target_rolling = is_rolling(case.get("notes") or "")
    # Practice and final targets are console launches (GTA VI is console-only).
    target_console = True
    target_pre = case_pre_peak_ratio(case, p1)
    donors = build_donors(case)
    bandwidth = size_bandwidth(target_ln1, donors)

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
            donors,
            hour,
            target_ln1,
            target_rolling,
            target_console,
            target_pre,
            bandwidth,
        )
        if logs is None:
            qs = baseline_quantiles(p1, hour)
        else:
            raw = []
            for log_ratio in logs:
                capped = min(math.log(MAX_VIEWERS / p1), max(-80.0, log_ratio))
                raw.append(p1 * math.exp(capped))
            qs = clean_quantiles(raw, p1)
        rows.append({"hour": hour, "q": qs})

    rows.sort(key=lambda r: r["hour"])
    if len(rows) != 23:
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
