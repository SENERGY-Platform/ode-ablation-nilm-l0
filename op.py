"""The operator: estimates the heat pump compressor's 15-minute mean power in W
from the site's main meter alone.

infer() buffers raw main-meter readings. When a message arrives in a new
15-minute interval, the interval LEAD_INTERVALS before the one that just closed
is estimated: its features include the intervals after it (see features.py),
and those are now complete. Every message of the following interval returns
that estimate stamped with the estimated interval's start, so each result
arrives 30 minutes after its interval ends and says which interval it is about.
"""

import datetime
import typing

import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger
from operator_lib.util.helpers.data import provide_historic_data_local
from operator_lib.util.logger import logger

from features import BUCKET, LEAD_INTERVALS, LOOKBACK_INTERVALS, bucket_aggregates, make_features
from training import train_model

# Raw history kept for the features: the longest look-back (LOOKBACK_INTERVALS,
# one week) measured from the estimated interval, plus margin for the counter
# difference that opens the oldest interval.
BUFFER = pd.Timedelta(BUCKET) * LOOKBACK_INTERVALS + pd.Timedelta(hours=2)
SEED_WINDOW = (BUFFER + pd.Timedelta(hours=1)).to_pytimedelta()
LEAD = pd.Timedelta(BUCKET) * LEAD_INTERVALS


class CustomConfig(Config):
    # Retrain at most this often, in seconds.
    retrain_after_s = 86400


def _utc_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _num(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


class Operator(MLOperator):
    configType = CustomConfig

    selectors = [
        Selector({"name": "main", "args": ["energy", "power"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self._main: typing.List[tuple] = []
        self._current_bucket: typing.Optional[pd.Timestamp] = None
        self._last_bucket: typing.Optional[pd.Timestamp] = None
        self._last_pred: typing.Optional[float] = None
        self._seeded = False
        super().init(*args, **kwargs)

    def _seed(self, now: pd.Timestamp):
        """Fill the buffer with the last week of main-meter history, so the
        first intervals after a start have their look-back features."""
        self._seeded = True
        try:
            datasets = provide_historic_data_local(SEED_WINDOW)
        except Exception as ex:  # a missing history must not stop inference
            logger.warning(f"could not seed the feature buffer: {ex}")
            return
        latest = None
        for ds in datasets:
            try:
                frame = ds.to_pandas()
            except Exception as ex:
                logger.warning(f"could not read a seed dataset: {ex}")
                continue
            if frame.empty or "time" not in frame.columns:
                continue
            if "energy" not in frame.columns and "power" not in frame.columns:
                continue
            times = pd.to_datetime(frame["time"], utc=True)
            e = frame["energy"] if "energy" in frame.columns else [np.nan] * len(frame)
            p = frame["power"] if "power" in frame.columns else [np.nan] * len(frame)
            for t, ev, pv in zip(times, e, p):
                if t < now:
                    self._main.append((t, _num(ev), _num(pv)))
                    latest = t if latest is None or t > latest else latest
        self._main.sort(key=lambda r: r[0])
        if latest is not None:
            self._current_bucket = latest.floor(BUCKET)

    def _finalize(self, model: PyFuncModel, closed: pd.Timestamp):
        """Estimate the interval LEAD_INTERVALS before the one that just closed."""
        self._last_bucket, self._last_pred = None, None
        target = closed - LEAD
        cutoff = target - BUFFER
        self._main = [r for r in self._main if r[0] >= cutoff]
        if not self._main:
            return
        main = pd.DataFrame(self._main, columns=["time", "energy", "power"])
        feats = make_features(bucket_aggregates(main))
        if target not in feats.index:
            return
        row = feats.loc[[target]]
        if pd.isna(row["p"].iloc[0]):
            return
        pred = float(np.asarray(model.predict(row)).ravel()[0])
        self._last_bucket, self._last_pred = target, pred

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        if model is None:
            return None, None, None
        now = _utc_ts(timestamp)
        if not self._seeded:
            self._seed(now)

        bucket = now.floor(BUCKET)
        if self._current_bucket is None:
            self._current_bucket = bucket
        elif bucket > self._current_bucket:
            try:
                self._finalize(model, self._current_bucket)
            except Exception as ex:
                logger.warning(f"could not estimate the interval before {self._current_bucket}: {ex}")
                self._last_bucket, self._last_pred = None, None
            self._current_bucket = bucket

        self._main.append((now, _num(data.get("energy")), _num(data.get("power"))))

        if self._last_bucket is None or self._last_pred is None:
            return None, None, None
        return self._last_bucket.to_pydatetime(), {"prediction": self._last_pred}, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
