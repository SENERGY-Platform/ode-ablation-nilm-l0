"""Feature construction, shared by training.py and op.py.

One code path for both sides on purpose: the operator buffers raw readings and
calls exactly these functions when a 15-minute interval closes, so a feature can
never mean one thing in training and another in inference.

Inputs are raw frames with a `time` column:
  main: energy (OBIS 1.8.0 import counter, raw kWh) and power (OBIS 16.7, raw W)
  cov:  cov_energy (heat pump heating-circuit counter, kWh)

The main meter is transformer-rated: every raw reading is multiplied by FACTOR.
"""

import numpy as np
import pandas as pd

FACTOR = 50.0                 # transformer ratio of the main meter
BUCKET = "15min"              # the resolution the operator estimates at
MAX_GAP_H = 2.0               # a counter difference across a longer gap is not a mean power
MAX_SITE_W = 5e6              # anything above this is a counter glitch, not load
LOCAL_TZ = "Europe/Berlin"

AGG_COLUMNS = ["p_counter", "pw_mean", "pw_min", "pw_max", "pw_std", "pw_n", "hz_p"]

FEATURES = [
    "p", "p_counter", "pw_mean", "pw_min", "pw_max", "pw_std", "pw_range", "pw_n",
    "p_lag1", "p_lag2", "p_lag3", "p_lag4", "p_lag8", "p_lag96",
    "p_d1", "p_d1_lag1",
    "p_mean4", "p_mean16", "p_mean96", "p_min96", "p_max96", "p_std4",
    "p_above_min96", "p_above_q10_96",
    "hz_p", "hz_lag1", "hz_lag2", "hz_mean4",
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


def bucket_aggregates(main, cov=None):
    """Per-bucket aggregates of the raw frames, indexed by UTC bucket start."""
    parts = []
    main = _clean(main, ["energy", "power"])
    pc = _counter_power(main, "energy", FACTOR * 1000.0)
    if len(pc):
        parts.append(pc.rename("p_counter"))

    pw = main.loc[main["power"].notna(), ["time", "power"]].copy()
    if not pw.empty:
        pw["power"] = pw["power"] * FACTOR
        pw["bucket"] = pw["time"].dt.floor(BUCKET)
        g = pw.groupby("bucket")["power"].agg(["mean", "min", "max", "std", "count"])
        g.columns = ["pw_mean", "pw_min", "pw_max", "pw_std", "pw_n"]
        parts.append(g)

    if cov is not None and len(cov):
        cov = _clean(cov, ["cov_energy"])
        hz = _counter_power(cov, "cov_energy", 1000.0)
        if len(hz):
            parts.append(hz.rename("hz_p"))

    if not parts:
        return pd.DataFrame(columns=AGG_COLUMNS)
    out = pd.concat(parts, axis=1).sort_index()
    for c in AGG_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[AGG_COLUMNS]


def make_features(agg):
    """Features per bucket, using only that bucket and earlier ones."""
    if agg is None or agg.empty:
        return pd.DataFrame(columns=FEATURES)
    idx = pd.date_range(agg.index.min(), agg.index.max(), freq=BUCKET)
    a = agg.reindex(idx).astype(float)
    f = pd.DataFrame(index=idx)

    p = a["p_counter"].fillna(a["pw_mean"])
    f["p"] = p
    for c in ["p_counter", "pw_mean", "pw_min", "pw_max", "pw_std", "pw_n"]:
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

    hz = a["hz_p"]
    f["hz_p"] = hz
    f["hz_lag1"] = hz.shift(1)
    f["hz_lag2"] = hz.shift(2)
    f["hz_mean4"] = hz.rolling(4, min_periods=1).mean()

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
