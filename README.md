# ode-ablation-nilm-l0

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

## Summary: approach and outcome

### The problem

Estimate the electrical power drawn by the heat pump's compressor, as a
15-minute mean in watts, from the site's main meter alone. The main meter is
transformer-rated with a factor of 50. The compressor has its own sub-meter,
which may be used for training and evaluation but is not available to the
operator when it runs. The criterion (`evaluation.yaml`, the developer's own) is
an MAE of at most 200 W over the test window, February 2026, with training on
history before 2026-02-01.

### Data

All real platform data; no simulation was needed. The data was found through
the platform ontology, working at exposure tier L0 throughout (availability and
volume only, no values).

| Role | Device | Series |
|---|---|---|
| Operator input | Hauptzähler (main meter) | `sensor.data.obis_1_8_0` (raw import counter, kWh, x50) and `sensor.data.obis_16_7` (instantaneous power, turned out empty) |
| Training label | Emonio Wärmepumpe Kompressor | `root.egy_watt` (training topic only, never delivered to `infer()`) |
| Training topic | Emonio Wärmepumpe Heizung | `root.egy_kwh` (read in training at the developer's request; not a feature, since the operator never receives it) |

Ruled out: the PV inverters and the second "Emonio Kompressor" (their history
ends in April 2025, before the test window), and the EV socket meter (its
message rate would have inflated the replay). Initially the heating-circuit
meter was proposed as a covariate. The developer then restricted the operator to
the main meter only, and it became a training topic.

### How the operator works

- **Features (`features.py`)**, one code path shared by training and inference:
  - 15-minute mean power from counter differences (x50);
  - within-interval statistics (mean, min, max, spread, step count) from
    consecutive counter readings about 30 s apart, used because OBIS 16.7 is
    empty for this meter;
  - lags and rolling baselines from one interval up to one week back;
  - the next two intervals (look-ahead);
  - local time of day and season.
- **Inference (`op.py`):**
  - buffers about a week of raw readings and seeds the buffer from history on
    start-up;
  - when an interval closes, estimates the interval two places back, so each
    estimate sees the switch-off as well as the switch-on;
  - returns that estimate stamped with the estimated interval's start, so each
    result arrives 30 minutes after its interval ends and says which interval
    it describes.
- **Training (`training.py`):**
  - gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`) on
    about 382 days of history;
  - October to April weighted three times, because the test month is in the
    heating season;
  - logs a 28-day validation MAE next to always-zero and median baselines.
  - Each history topic is read on its own, so a failed read is logged and
    named instead of killing the run unexplained.
- Every change to features or inference was checked first on synthetic data in
  the developer's pod. The check confirms that the operator's
  interval-by-interval path reproduces the training features exactly.

### What was learnt on the way

- **Absolute-error loss did not work.** It looked like the natural loss for an
  MAE criterion, but the booster stayed at an all-zero prediction: the
  compressor's median 15-minute power is exactly 0, and the leaf medians never
  move. The synthetic check caught this before any launch. Squared-error loss
  is used instead.
- **Topic names.** Device-service topic names keep the hyphens of the service
  id (`urn_infai_ses_service_9dc0f355-85ef-...`). The first two launches used
  underscores, so every history read requested a non-existent service and
  failed. The per-topic read diagnostics made this visible.
- **Target collisions.** The evaluation matches its target by path alone. All
  Emonio meters share one service, so `root.egy_watt` is mapped only on the
  compressor. Any other Emonio device is mapped through a different variable.

### Results (February 2026, MAE against the 200 W threshold)

| Run | Change | February MAE | Validation MAE (Jan) |
|---|---|---|---|
| `umqy2ng` | counter differences only | 953 W | 1077 W |
| `iwgxg6e` | + within-interval statistics from consecutive readings | 913 W | 1022 W |
| `qqwwnxd` | + 30-minute look-ahead | 881 W | 1002 W |
| `y3tuhfz` | + one-week look-back, heating-season weight 3 | **872 W** | 982 W |
| `3fw55ic` | energy from `obis_1_8_0_f` instead | 900 W | 1009 W |

The always-zero baseline is about 1877 W, which is also roughly the
compressor's mean draw in January. The best model removes about 48% of that
error.

**The criterion is not met.** The best February MAE is more than four times the
threshold, and the gains shrank with each change.

The most likely limit is the main meter's counter resolution. A raw step of
0.001 kWh becomes 50 Wh after the factor of 50, which is about 200 W of rounding
in every 15-minute mean. This is unverified, because it needs a value-level
profile (tier L1) that this session never had. The `_f` counter variant was
tested as a possibly more precise reading, and it did not help.

The repository and the session selection are left in the best-scoring
configuration: energy from `obis_1_8_0`, code at or after `f484a71`.

### Open questions for whoever continues

1. **Profile the counter.** `profile_series` on `sensor.data.obis_1_8_0` (tier
   L1) would show whether its step size makes 200 W unreachable from this meter.
2. **Inputs beyond the main meter.** Adding inputs, for example outdoor
   temperature or the heating-circuit meter, would let the model see what
   drives the compressor. That would reverse the main-meter-only decision.
3. **The criterion.** Whether an absolute 200 W suits this signal is the
   developer's call.
4. **Pipeline mappings.** When deploying, map the operator's `energy` input to
   `sensor.data.obis_1_8_0` and `power` to `sensor.data.obis_16_7`.
   `operator.yaml` names the inputs but not their source paths.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "features.py" | Feature construction shared by training and inference. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.9.0". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-nilm-l0". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-nilm-l0:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.9.0", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
