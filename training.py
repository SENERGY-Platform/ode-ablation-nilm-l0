"""Training, on Ray.

The main meter is the only input topic. The compressor sub-meter (the label) and
the heat pump heating-circuit counter are training topics: read here, never
delivered to infer(). The heating-circuit counter is not a feature -- a feature
has to exist when the operator runs, and that meter will not -- so it is read
but unused at present.

Each topic's read is collected on its own. A topic whose read fails is logged as
a metric read_failed_<role>_<dests> and skipped; only a missing main meter or a
missing label stops training, and then the error names which topic failed.

Squared-error loss, although the criterion is MAE: with absolute-error loss the
booster starts from the target's median, which is exactly 0 for a compressor
that is off much of the time, and its leaf medians stay at 0 -- it never leaves
the all-zero prediction (checked on synthetic on/off data before the first run).

The last VALIDATION before the training end is held out once for a validation
MAE that is logged on the run; the model returned is then refitted on all of it.
"""

import datetime
import typing

import numpy as np
import pandas as pd
import ray
from mlflow.pyfunc import PythonModel
from sklearn.ensemble import HistGradientBoostingRegressor

import operator_lib.util as util
from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger
from operator_lib.util.logger import logger as lib_logger

from features import FEATURES, bucket_aggregates, bucket_target, make_features


# The main meter starts 2025-01-16; this reaches back to it. All topics are
# device series read from timescale, so the 365-day Kafka bound does not apply.
TRAINING_WINDOW = datetime.timedelta(days=382)
VALIDATION = datetime.timedelta(days=28)

HGB_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=600,
    max_leaf_nodes=63,
    min_samples_leaf=40,
    l2_regularization=1.0,
    random_state=0,
)


class CompressorPowerModel(PythonModel):
    """Maps one row of features.FEATURES to the compressor's mean power in W."""

    def __init__(self, estimator, features: typing.List[str]) -> None:
        self.estimator = estimator
        self.features = list(features)

    def predict(self, context, model_input=None, params=None):
        frame = model_input if model_input is not None else context
        X = pd.DataFrame(frame)[self.features].astype(float)
        return np.clip(self.estimator.predict(X), 0.0, None)


def _topic_labels(role: str) -> typing.List[str]:
    """One label per topic of a role, in the order provide_historic_data
    returns them: the role and the topic's mapping dests, as plain words."""
    try:
        dep_config = util.DeploymentConfig()
        opr_config = util.OperatorConfig(util.load_operator_config_json(dep_config))
        topics = opr_config.trainingTopics if role == "training" else opr_config.inputTopics
        return [f"{role}_" + "_".join(m.dest for m in t.mappings) for t in topics]
    except Exception:
        return []


def _frames(refs, role: str) -> typing.Tuple[typing.List[pd.DataFrame], typing.List[str]]:
    """Collect each topic's read separately. Returns the frames that arrived and
    the labels of the topics whose read failed."""
    labels = _topic_labels(role)
    frames, failed = [], []
    for i, ref in enumerate(refs):
        label = labels[i] if i < len(labels) else f"{role}_topic_{i}"
        try:
            ds = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
            frames.append(ds.to_pandas())
        except Exception as ex:
            lib_logger.error(f"history read failed for {label}: {type(ex).__name__}: {ex}")
            failed.append(f"{label} ({type(ex).__name__})")
    return frames, failed


def _pick(frames, column) -> typing.Optional[pd.DataFrame]:
    for frame in frames:
        if column in frame.columns:
            return frame
    return None


def build_dataset(main, target) -> pd.DataFrame:
    feats = make_features(bucket_aggregates(main))
    y = bucket_target(target)
    data = feats.join(y, how="inner")
    return data[data["target"].notna() & data["p"].notna()]


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    with logger.trace("read history"):
        inputs, failed_inputs = _frames(provide_historic_data(TRAINING_WINDOW), "input")
        labels, failed_labels = _frames(
            provide_historic_data(TRAINING_WINDOW, role="training"), "training")

    failed = failed_inputs + failed_labels
    if failed:
        logger.log_metrics({
            "read_failed_" + f.split(" ")[0]: 1.0 for f in failed
        })

    main = _pick(inputs, "energy")
    target = _pick(labels, "target")
    if main is None or target is None:
        raise ValueError(
            "required history missing: main meter "
            f"{'read' if main is not None else 'NOT read'}, label "
            f"{'read' if target is not None else 'NOT read'}; "
            f"failed topics: {', '.join(failed) if failed else 'none (a topic returned no frame)'}")

    with logger.trace("features"):
        data = build_dataset(main, target)
    if data.empty:
        raise ValueError(
            f"no 15-minute interval has both a main-meter power and a target: "
            f"main rows {len(main)}, target rows {len(target)}")

    cutoff = data.index.max() - VALIDATION
    fit_part, val_part = data[data.index < cutoff], data[data.index >= cutoff]

    metrics = {
        "n_intervals": float(len(data)),
        "n_fit": float(len(fit_part)),
        "n_val": float(len(val_part)),
        "coverage_p_counter": float(data["p_counter"].notna().mean()),
        "coverage_pw_mean": float(data["pw_mean"].notna().mean()),
        "read_failed_topics": float(len(failed)),
    }

    with logger.trace("validate"):
        if len(fit_part) > 500 and len(val_part) > 50:
            est = HistGradientBoostingRegressor(**HGB_PARAMS)
            est.fit(fit_part[FEATURES].astype(float), fit_part["target"])
            pred = np.clip(est.predict(val_part[FEATURES].astype(float)), 0.0, None)
            y = val_part["target"].to_numpy()
            metrics["val_mae"] = float(np.mean(np.abs(pred - y)))
            metrics["val_mae_zero"] = float(np.mean(np.abs(y)))
            metrics["val_mae_median"] = float(np.mean(np.abs(y - np.median(fit_part["target"]))))

    with logger.trace("fit"):
        est = HistGradientBoostingRegressor(**HGB_PARAMS)
        est.fit(data[FEATURES].astype(float), data["target"])

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "validation_days": VALIDATION.days,
        **{f"hgb_{k}": v for k, v in HGB_PARAMS.items()},
    })
    logger.log_metrics(metrics)
    return CompressorPowerModel(est, FEATURES)
