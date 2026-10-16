"""Feature construction, shared by training.py and op.py.

One code path for both sides on purpose: the operator buffers raw readings and
calls exactly these functions when a 15-minute interval closes, so a feature can
never mean one thing in training and another in inference.

The only input is the main meter, as a raw frame with a `time` column:
  energy  import counter, raw kWh (OBIS 1.8.0; the launch decides which variant)
  power   OBIS 16.7 instantaneous power, raw W

The main meter is transformer-rated: every raw reading is multiplied by FACTOR.
Whether the obis_1_8_0_f variant is already multiplied is not declared; the
model is insensitive to a constant scale, and MAX_SITE_W is set high enough
that a doubly multiplied value is kept rather than discarded as a glitch.

OBIS 16.7 turned out to be empty for this meter (run umqy2ngstnlnuedubwlz52uy54
logged coverage_pw_mean = 0). The within-interval statistics are therefore also
derived from consecutive counter readings -- the meter publishes about every
30 seconds -- and used wherever a measured power is absent.

Look-ahead: the features of interval k include the two intervals after it
(LEAD_INTERVALS). The operator therefore estimates interval k once interval
k+2 has closed, i.e. 30 minutes late, and stamps the estimate with k's start.

Look-back: up to one week (LOOKBACK_INTERVALS), so the model sees the site's
weekly level and the same time a week earlier, which carry the slow heating
demand that drives the compressor's duty cycle.
"""

import numpy as np
import pandas as pd

FACTOR = 50.0                 # transformer ratio of the main meter
BUCKET = "15min"              # the resolution the operator estimates at
LEAD_INTERVALS = 2            # how many later intervals an estimate waits for
LOOKBACK_INTERVALS = 672      # the longest look-back of any feature: one week
MAX_GAP_H = 2.0               # a counter difference across a longer gap is not a mean power
MAX_READING_GAP_H = 5.0 / 60  # consecutive readings further apart give no short-interval power
STEP_W = 1000.0               # a change between consecutive powers larger than this is a step
MAX_SITE_W = 5e8              # above this is a counter glitch; high enough for a double factor
LOCAL_TZ = "Europe/Berlin"

AGG_COLUMNS = ["p_counter", "pw_mean", "pw_min", "pw_max", "pw_std", "pw_n", "pw_steps"]

FEATURES = [
    "p", "p_counter", "pw_mean", "pw_min", "pw_max", "pw_std", "pw_range", "pw_n", "pw_steps",
    "p_lag1", "p_lag2", "p_lag3", "p_lag4", "p_lag8", "p_lag96",
    "p_d1", "p_d1_lag1",
    "p_mean4", "p_mean16", "p_mean96", "p_min96", "p_max96", "p_std4",
    "p_above_min96", "p_above_q10_96",
    "p_mean672", "p_lag672", "p_mean96_lag96",
    "p_lead1", "p_lead2", "pw_steps_lead1", "pw_range_lead1",
    "hour", "weekday", "doy_sin", "doy_cos",
]


def _utc(values):
    return pd.to_datetime(values, utc=True)


def _clean(df, cols):
    df = df.copy()
    df["time"] = _utc(df["time"])
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    return df.sort_values("time", kind="stable")


def _counter_power(df, col, wh_per_unit):
    """Mean power per bucket from a cumulative counter, in W.

    The rate between the last reading of the previous bucket and the last
    reading of this one. A decreasing counter, a gap over MAX_GAP_H or an
    implausible rate gives NaN rather than a number.
    """
    d = df.loc[df[col].notna() & np.isfinite(df[col]), ["time", col]].copy()
    if d.empty:
        return pd.Series(dtype=float)
    d["bucket"] = d["time"].dt.floor(BUCKET)
    last = d.groupby("bucket").agg(t=("time", "last"), v=(col, "last"))
    dv = last["v"].diff()
    dt_h = last["t"].diff().dt.total_seconds() / 3600.0
    p = dv * wh_per_unit / dt_h
    bad = (dv < 0) | (dt_h <= 0) | (dt_h > MAX_GAP_H) | (p > MAX_SITE_W)
    return p.mask(bad)


def _reading_power(main):
    """Power between each pair of consecutive counter readings, in W, stamped
    with the later reading. Pairs more than MAX_READING_GAP_H apart, a
    decreasing counter or an implausible rate are dropped."""
    d = main.loc[main["energy"].notna() & np.isfinite(main["energy"]), ["time", "energy"]]
    if len(d) < 2:
        return pd.DataFrame(columns=["time", "power"])
    dt_h = d["time"].diff().dt.total_seconds() / 3600.0
    de = d["energy"].diff()
    p = de * FACTOR * 1000.0 / dt_h
    ok = (dt_h > 0) & (dt_h <= MAX_READING_GAP_H) & (de >= 0) & (p <= MAX_SITE_W)
    return pd.DataFrame({"time": d["time"], "power": p})[ok.fillna(False)]


def _power_stats(pw):
    """Within-bucket statistics of a power series with columns time, power (W)."""
    if pw.empty:
        return None
    pw = pw.sort_values("time", kind="stable").copy()
    pw["step"] = (pw["power"].diff().abs() > STEP_W).astype(float)
    pw["bucket"] = pw["time"].dt.floor(BUCKET)
    return pw.groupby("bucket").agg(
        pw_mean=("power", "mean"),
        pw_min=("power", "min"),
        pw_max=("power", "max"),
        pw_std=("power", "std"),
        pw_n=("power", "count"),
        pw_steps=("step", "sum"),
    )


def bucket_aggregates(main):
    """Per-bucket aggregates of the raw main-meter frame, indexed by UTC bucket start."""
    parts = []
    main = _clean(main, ["energy", "power"])
    pc = _counter_power(main, "energy", FACTOR * 1000.0)
    if len(pc):
        parts.append(pc.rename("p_counter"))

    measured = main.loc[main["power"].notna(), ["time", "power"]].copy()
    measured["power"] = measured["power"] * FACTOR
    stats_measured = _power_stats(measured)
    stats_derived = _power_stats(_reading_power(main))
    if stats_measured is not None and stats_derived is not None:
        parts.append(stats_measured.combine_first(stats_derived))
    elif stats_measured is not None:
        parts.append(stats_measured)
    elif stats_derived is not None:
        parts.append(stats_derived)

    if not parts:
        return pd.DataFrame(columns=AGG_COLUMNS)
    out = pd.concat(parts, axis=1).sort_index()
    for c in AGG_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[AGG_COLUMNS]


def make_features(agg):
    """Features per bucket: that bucket, up to a week before it, and LEAD_INTERVALS after it."""
    if agg is None or agg.empty:
        return pd.DataFrame(columns=FEATURES)
    idx = pd.date_range(agg.index.min(), agg.index.max(), freq=BUCKET)
    a = agg.reindex(idx).astype(float)
    f = pd.DataFrame(index=idx)

    p = a["p_counter"].fillna(a["pw_mean"])
    f["p"] = p
    for c in ["p_counter", "pw_mean", "pw_min", "pw_max", "pw_std", "pw_n", "pw_steps"]:
        f[c] = a[c]
    f["pw_range"] = a["pw_max"] - a["pw_min"]

    for k in (1, 2, 3, 4, 8, 96):
        f[f"p_lag{k}"] = p.shift(k)
    f["p_d1"] = p - p.shift(1)
    f["p_d1_lag1"] = p.shift(1) - p.shift(2)
    f["p_mean4"] = p.rolling(4, min_periods=1).mean()
    f["p_mean16"] = p.rolling(16, min_periods=1).mean()
    f["p_mean96"] = p.rolling(96, min_periods=1).mean()
    f["p_min96"] = p.rolling(96, min_periods=1).min()
    f["p_max96"] = p.rolling(96, min_periods=1).max()
    f["p_std4"] = p.rolling(4, min_periods=2).std()
    f["p_above_min96"] = p - f["p_min96"]
    f["p_above_q10_96"] = p - p.rolling(96, min_periods=1).quantile(0.1)

    f["p_mean672"] = p.rolling(LOOKBACK_INTERVALS, min_periods=1).mean()
    f["p_lag672"] = p.shift(LOOKBACK_INTERVALS)
    f["p_mean96_lag96"] = f["p_mean96"].shift(96)

    f["p_lead1"] = p.shift(-1)
    f["p_lead2"] = p.shift(-2)
    f["pw_steps_lead1"] = f["pw_steps"].shift(-1)
    f["pw_range_lead1"] = f["pw_range"].shift(-1)

    local = idx.tz_convert(LOCAL_TZ)
    f["hour"] = local.hour + local.minute / 60.0
    f["weekday"] = local.weekday
    doy = local.dayofyear.to_numpy(dtype=float)
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f[FEATURES]


def bucket_target(df):
    """Mean of the target per bucket, which is exactly what the scorer compares with."""
    d = _clean(df, ["target"])
    d = d[d["target"].notna()].copy()
    if d.empty:
        return pd.Series(dtype=float, name="target")
    d["bucket"] = d["time"].dt.floor(BUCKET)
    return d.groupby("bucket")["target"].mean().rename("target")
