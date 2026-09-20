"""Calibrated Fusion: seven signals into one explainable probability.

WHY THIS IS NOT A NEURAL NETWORK

MailGuard deliberately does not use one large black box classifier over
raw mail. It uses seven independent scorers and a transparent layer that
combines them, so that every point of the final score traces back to a
specific finding an analyst can read, argue with, and put in front of a
bank or a court.

The fusion model is an Explainable Boosting Machine, a generalised
additive model with pairwise interaction terms:

    P(fraud) = sigmoid( f0 + sum_i f_i(x_i) + sum_ij f_ij(x_i, x_j) )

Each f_i is a learned shape function over one signal's score, so the
contribution of that signal is separable and printable: "identity added
3.3 points, intent added 2.8, the two together added another 1.7". A deep
network over the same inputs would score marginally better on a held out
set and would be impossible to decompose. A verdict you cannot explain is
not evidence, and evidence is the entire product. That trade is made
once, here, on purpose.

TWO BEHAVIOURS THAT MATTER MORE THAN THE WEIGHTS

1. Missing data is not zero. A signal that abstained could not run; a
   signal that scored 0.0 ran and found nothing. The feature vector
   therefore carries fourteen values, seven scores and seven presence
   flags, so the model learns what an absence means instead of this code
   asserting that it means nothing. A brand new sender with no baseline
   is exactly the profile an attacker presents, and scoring that as a
   clean neutral would reward the attack.

2. Contributions are real numbers in log-odds points, and they sum
   exactly to the logit. Nothing in this module invents an attribution
   for display. If the trained model cannot produce local attributions,
   this module does not use its probability either.
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterable, Optional

if __package__ in (None, ""):  # pragma: no cover - script convenience only
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from mailguard.core.models import ParsedEmail, SignalResult, Verdict
from mailguard.fusion.verdict import to_action, to_verdict_class

# ----------------------------------------------------------------------
# Optional dependencies, guarded. Neither is needed for the fallback.
# ----------------------------------------------------------------------
try:
    import interpret  # noqa: F401  (presence check only)

    HAS_INTERPRET = True
except ImportError:
    HAS_INTERPRET = False

try:
    import joblib  # noqa: F401  (presence check only)

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# ======================================================================
# TRAINED MODEL SLOT
# ----------------------------------------------------------------------
# The model for this stage has NOT been trained yet.
# When it is, put the artefact path in MODEL_PATH below and the code
# will pick it up automatically on the next run. Until then the
# documented heuristic in _heuristic_logit() runs instead, so the whole
# pipeline stays runnable end to end today.
#
# Expected artefact: a joblib dumped interpret.glassbox
# ExplainableBoostingClassifier, fitted on a 14 column matrix in the
# order given by FEATURE_NAMES below, with the positive class meaning
# "fraudulent". Keep interactions on (interactions=10 or so): the x2/x5
# pair is the one that matters and the model should be allowed to learn
# its shape rather than being handed the placeholder product term used
# here.
#
# The model must be able to produce local explanations
# (ebm.explain_local(X)), because this module refuses to publish a
# probability it cannot decompose. See _model_score_and_contributions().
# Expected feature order is defined by FEATURE_NAMES below.
# ======================================================================
MODEL_PATH: str = ""          # <-- trained model path goes here

# The fixed signal order. Everything downstream, including the trained
# model's column order, depends on this constant.
SIGNAL_ORDER: list[str] = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]

# Fourteen features: seven scores, then seven missingness indicators.
FEATURE_NAMES: list[str] = [f"{sid}_score" for sid in SIGNAL_ORDER] + [
    f"{sid}_is_present" for sid in SIGNAL_ORDER
]

# Human names, for the report and for the shape function plots the
# trained model will produce.
SIGNAL_LABELS: dict[str, str] = {
    "x1": "Authentication",
    "x2": "Identity",
    "x3": "Header and Route",
    "x4": "Sender Baseline",
    "x5": "Intent",
    "x6": "Origin and Infrastructure",
    "x7": "Attachment and URL",
}

# Statuses that mean "this signal did not produce a usable score".
UNUSABLE_STATUSES: frozenset[str] = frozenset({"abstain", "error", "skipped", "unavailable"})

# ----------------------------------------------------------------------
# PLACEHOLDER PRIORS. These are not fitted parameters. They are the
# weights of a logistic combination standing in for the EBM's shape
# functions until it is trained, and every one of them should be replaced
# by a learned f_i.
#
# The relative ordering encodes what the team actually believes about
# these signals:
#
#   x2 (identity) and x5 (intent) carry the most signal. Between them
#   they cover business email compromise and commodity phishing, which is
#   the bulk of real loss.
#
#   x1 (authentication) carries almost nothing on its own. This is the
#   counterintuitive one and it is deliberate: an attacker who registers
#   their own lookalike domain publishes correct SPF and DKIM records and
#   authenticates perfectly. SPF and DKIM prove custody of a domain, not
#   honesty of a sender. x1 is kept in the model because a FAILING
#   authentication on a mail that claims to be a bank is still worth a
#   fraction of a point, and because its absence is informative.
#
#   x7 (payload) is high because a macro document or an executable is
#   close to dispositive on its own.
#
#   x4 (baseline) is moderate but matters enormously in the one case no
#   other signal covers: a compromised legitimate account.
# ----------------------------------------------------------------------
SIGNAL_WEIGHTS: dict[str, float] = {
    "x1": 0.9,   # authentication: weak, for the reason above
    "x2": 3.6,   # identity: carries BEC
    "x3": 1.6,   # header and route anomalies
    "x4": 1.8,   # sender baseline: the compromised-account case
    "x5": 3.2,   # intent: carries commodity phishing
    "x6": 1.4,   # origin and infrastructure
    "x7": 3.4,   # payload: a live macro or executable is near dispositive
}

# A property worth stating because it is a design decision and not an
# accident of the numbers above: with this intercept, no single signal at
# the top of its range reaches the BLOCK band on its own. A maxed out
# lone signal is indistinguishable from a broken signal, so BLOCK
# requires corroboration - two strong findings, or one strong finding
# plus its interaction term. Single maxima land in WARN, where an analyst
# sees them and the mail still reaches the user.

# Pairwise interaction priors, standing in for the EBM's f_ij terms. Only
# pairs with a mechanism behind them are included; an interaction without
# a story is an overfit waiting to happen.
INTERACTION_WEIGHTS: dict[tuple[str, str], float] = {
    # A forged identity asking for money is the whole of BEC. This is the
    # single most important term in the model.
    ("x2", "x5"): 2.2,
    # A lookalike domain shipping a payload is a prepared campaign.
    ("x2", "x7"): 0.9,
    # A sender writing unlike themselves and asking for something new is
    # the shape of an account takeover.
    ("x4", "x5"): 0.8,
    # Hostile infrastructure carrying a hostile payload.
    ("x6", "x7"): 0.6,
}

# What an ABSENCE is worth, in log-odds points. These are the numbers the
# EBM will learn from the is_present columns; until then they are set
# from the same argument the whole design rests on.
#
# Every one is positive, which is the point: an unmeasurable signal is
# mildly adverse, not neutral. An attacker's profile is precisely the one
# that cannot be measured - no history, no prior thread, no baseline. The
# values are small (0.25 to 0.6) so that absence alone can never reach a
# verdict on its own: seven abstentions plus the intercept sits at about
# p = 0.20, which is PASS. That is correct. Not being able to measure
# anything is not evidence of fraud, it is just not evidence of
# innocence either.
MISSING_WEIGHTS: dict[str, float] = {
    "x1": 0.35,  # no authentication result to read
    "x2": 0.50,  # no identity assessment at all
    "x3": 0.30,  # no route to analyse
    "x4": 0.60,  # no sender baseline: the attacker's default state
    "x5": 0.40,  # no readable content
    "x6": 0.25,  # origin unresolvable
    "x7": 0.30,  # no payload to inspect
}

# Base rate term. -4.2 log-odds is about 1.5 percent, which is the right
# order for the fraudulent share of mail that has already survived an
# upstream spam filter. Refit this per deployment: a finance team's inbox
# and a public support address do not share a base rate.
INTERCEPT: float = -4.2

# Key used for the intercept inside the contributions dict, so that the
# dict sums exactly to the logit.
INTERCEPT_KEY: str = "_intercept"

# The reported probability is clamped away from 0 and 1. A saturated
# additive model reaches sigmoid values that round to exactly 1.0000, and
# printing that in a forensic report is a claim of certainty this tool is
# not entitled to make: it says no evidence could ever change the answer,
# which is both false and the first thing a defence expert will attack.
# The itemised log-odds in `contributions` is the unclamped number, so
# nothing is lost, and the clamp only bites past p = 0.999999 either way.
PROBABILITY_FLOOR: float = 1e-6
PROBABILITY_CEILING: float = 1.0 - 1e-6

_MODEL: Any = None
_MODEL_TRIED: bool = False
_CALIBRATOR: Any = None
_CALIBRATOR_TRIED: bool = False

# ======================================================================
# CALIBRATOR SLOT
# ----------------------------------------------------------------------
# Expected artefact: a joblib dumped isotonic regression
# (sklearn.isotonic.IsotonicRegression) or Platt scaler exposing
# predict(X) or predict_proba(X), fitted on held out mail so that the
# output means what it says. Default is the identity function.
# ======================================================================
CALIBRATOR_PATH: str = ""     # <-- trained calibrator path goes here


def _sigmoid(value: float) -> float:
    """Numerically safe logistic function."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exponential = math.exp(max(value, -60.0))
    return exponential / (1.0 + exponential)


def calibrate(raw_score: float) -> float:
    """Map a raw model score to a probability that means what it says.

    Calibration is what makes 0.9 mean roughly nine in ten. Without it,
    a model's output is only an ordering: it will rank a fraud above a
    newsletter perfectly well while its numeric 0.9 corresponds to a true
    rate of 0.6, and every threshold set on that number is then wrong by
    an unknown amount. Since the three action bands in this system are
    defined as probability cut points, and since the false positive
    arithmetic behind them is stated in probabilities, an uncalibrated
    score would make the whole policy meaningless.

    Defaults to the identity function, so the untrained pipeline is
    honest about doing no calibration rather than pretending to.
    """
    global _CALIBRATOR, _CALIBRATOR_TRIED
    value = max(0.0, min(1.0, float(raw_score)))
    if not _CALIBRATOR_TRIED:
        _CALIBRATOR_TRIED = True
        if CALIBRATOR_PATH and os.path.exists(CALIBRATOR_PATH) and HAS_JOBLIB:
            try:
                import joblib

                _CALIBRATOR = joblib.load(CALIBRATOR_PATH)
            except Exception:
                _CALIBRATOR = None
    if _CALIBRATOR is None:
        return value
    try:
        if hasattr(_CALIBRATOR, "predict_proba"):
            return max(0.0, min(1.0, float(_CALIBRATOR.predict_proba([[value]])[0][1])))
        return max(0.0, min(1.0, float(_CALIBRATOR.predict([value])[0])))
    except Exception:
        return value


# ----------------------------------------------------------------------
# Feature vector
# ----------------------------------------------------------------------
def is_usable(signal: Optional[SignalResult]) -> bool:
    """True when a signal produced a score fusion is allowed to use.

    Anything other than a clean status is treated as missing rather than
    as a low score. Being permissive here is deliberate: if a signal
    reports a status this module has never heard of, the safe reading is
    "I cannot interpret this", not "this must be fine".
    """
    if signal is None:
        return False
    status = (signal.status or "").lower()
    if status in UNUSABLE_STATUSES:
        return False
    if status not in ("ok", "", "pass", "scored"):
        return False
    try:
        float(signal.score)
    except (TypeError, ValueError):
        return False
    return True


def build_feature_vector(
    signals: Iterable[SignalResult],
) -> tuple[list[float], dict[str, Optional[SignalResult]]]:
    """Build the fourteen value feature vector in SIGNAL_ORDER.

    Returns (vector, by_id). The vector is seven scores followed by seven
    is_present flags. A signal that is absent from the input list or that
    abstained contributes a score of 0.0 AND a presence flag of 0.0, and
    the model reads the pair, never the score alone. That is the whole
    mechanism by which "could not measure" stays distinct from
    "measured, nothing found".
    """
    by_id: dict[str, Optional[SignalResult]] = {sid: None for sid in SIGNAL_ORDER}
    for signal in signals or []:
        if signal is None:
            continue
        if signal.signal_id in by_id and by_id[signal.signal_id] is None:
            by_id[signal.signal_id] = signal

    scores: list[float] = []
    presence: list[float] = []
    for sid in SIGNAL_ORDER:
        signal = by_id[sid]
        if is_usable(signal):
            assert signal is not None  # narrowed by is_usable
            scores.append(max(0.0, min(1.0, float(signal.score))))
            presence.append(1.0)
        else:
            scores.append(0.0)
            presence.append(0.0)
    return scores + presence, by_id


# ----------------------------------------------------------------------
# Heuristic fallback
# ----------------------------------------------------------------------
def _heuristic_logit(vector: list[float]) -> tuple[float, dict[str, float]]:
    """Weighted logistic combination standing in for the fitted EBM.

    Returns (logit, contributions in log-odds points). The contributions
    dict holds one entry per signal, one per active interaction term, and
    the intercept, and it sums exactly to the logit so the report can
    show the arithmetic rather than assert it.

    For a present signal the contribution is weight * score, the linear
    stand-in for the shape function f_i. For an absent one it is the
    missingness prior from MISSING_WEIGHTS, which is the stand-in for
    what the EBM will learn from the is_present column.
    """
    count = len(SIGNAL_ORDER)
    scores = dict(zip(SIGNAL_ORDER, vector[:count]))
    presence = dict(zip(SIGNAL_ORDER, vector[count : 2 * count]))

    contributions: dict[str, float] = {INTERCEPT_KEY: INTERCEPT}
    for sid in SIGNAL_ORDER:
        if presence[sid] >= 0.5:
            contributions[sid] = SIGNAL_WEIGHTS[sid] * scores[sid]
        else:
            contributions[sid] = MISSING_WEIGHTS[sid]

    for (first, second), weight in INTERACTION_WEIGHTS.items():
        # An interaction between a measured and an unmeasured signal is
        # not knowable, so it is left out rather than guessed at.
        if presence.get(first, 0.0) >= 0.5 and presence.get(second, 0.0) >= 0.5:
            term = weight * scores[first] * scores[second]
            if term:
                contributions[f"{first}*{second}"] = term

    return sum(contributions.values()), contributions


# ----------------------------------------------------------------------
# Trained model path
# ----------------------------------------------------------------------
def _load_model() -> Any:
    """Load the trained EBM once, or return None if the slot is empty."""
    global _MODEL, _MODEL_TRIED
    if _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        return None
    if not HAS_JOBLIB:
        return None
    try:
        import joblib

        _MODEL = joblib.load(MODEL_PATH)
    except Exception:
        _MODEL = None
    return _MODEL


def _term_to_key(term_name: str) -> Optional[str]:
    """Map an EBM term name back to a contributions key.

    EBM term names come from the training column names, so
    "x2_score" maps to "x2", "x4_is_present" maps to "x4", and an
    interaction term "x2_score & x5_score" maps to "x2*x5".
    """
    cleaned = term_name.strip()
    parts = [part.strip() for part in cleaned.split("&")]
    ids: list[str] = []
    for part in parts:
        for sid in SIGNAL_ORDER:
            if part.startswith(sid + "_") or part == sid:
                ids.append(sid)
                break
    if not ids:
        return None
    if len(ids) == 1:
        return ids[0]
    return "*".join(dict.fromkeys(ids))


def _model_score_and_contributions(
    vector: list[float],
) -> Optional[tuple[float, dict[str, float]]]:
    """Score with the trained EBM, with real per-term attributions.

    Returns (probability, contributions in log-odds points) or None.

    If the model scores but its local explanation cannot be read, this
    function returns None and the caller falls back to the heuristic for
    BOTH the probability and the contributions. That is deliberate: a
    probability without a decomposition is exactly the black box output
    this architecture exists to avoid, and mixing a model's probability
    with the heuristic's attribution would print an evidence table whose
    numbers do not add up to the verdict above it.
    """
    model = _load_model()
    if model is None:
        return None
    try:
        matrix = [list(vector)]
        probability = float(model.predict_proba(matrix)[0][1])

        explanation = model.explain_local(matrix)
        data = explanation.data(0)
        names = list(data.get("names") or [])
        values = [float(v) for v in (data.get("scores") or [])]
        if not names or len(names) != len(values):
            return None

        contributions: dict[str, float] = {}
        for name, value in zip(names, values):
            key = _term_to_key(str(name))
            if key is None:
                continue
            contributions[key] = contributions.get(key, 0.0) + value

        intercept = data.get("extra", {}).get("scores")
        if isinstance(intercept, (list, tuple)) and intercept:
            contributions[INTERCEPT_KEY] = float(intercept[0])
        else:
            contributions[INTERCEPT_KEY] = float(getattr(model, "intercept_", [0.0])[0])

        # Keep the printed arithmetic honest: the contributions must
        # reconstruct the model's own logit. If they do not, we do not
        # trust the explanation and fall back.
        logit = sum(contributions.values())
        expected = math.log(max(probability, 1e-12) / max(1.0 - probability, 1e-12))
        if abs(logit - expected) > 0.25:
            return None
        return probability, contributions
    except Exception:
        return None


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def fuse(signals: list[SignalResult], email: ParsedEmail) -> Verdict:
    """Fuse seven signal scores into one calibrated, explainable verdict.

    The returned Verdict carries:

      probability     calibrated P(fraud)
      verdict_class   one of five classes, see fusion.verdict
      action          BLOCK, WARN or PASS
      contributions   log-odds points per term. Keys are the seven signal
                      ids, plus one key per active interaction ("x2*x5"),
                      plus "_intercept". They sum to the logit, so an
                      analyst can check the arithmetic of the verdict by
                      adding up the evidence table.
      signals         the input list, unmodified, including abstentions

    campaign_id and evidence_hash are left None here; the campaign graph
    and the evidence ledger fill them in, because neither is a property
    of the fusion arithmetic.
    """
    vector, _by_id = build_feature_vector(signals)

    model_result = _model_score_and_contributions(vector)
    if model_result is not None:
        raw_probability, contributions = model_result
    else:
        logit, contributions = _heuristic_logit(vector)
        raw_probability = _sigmoid(logit)

    probability = min(max(calibrate(raw_probability), PROBABILITY_FLOOR), PROBABILITY_CEILING)
    signal_list = list(signals or [])
    return Verdict(
        probability=round(float(probability), 6),
        verdict_class=to_verdict_class(probability, signal_list),
        action=to_action(probability),
        contributions={key: round(float(value), 6) for key, value in contributions.items()},
        signals=signal_list,
        attribution=email.attribution if email is not None else None,
        campaign_id=None,
        evidence_hash=None,
    )


def signal_contributions(contributions: dict[str, float]) -> dict[str, float]:
    """Just the seven per-signal terms, dropping interactions and intercept."""
    return {key: value for key, value in contributions.items() if key in SIGNAL_ORDER}


def interaction_contributions(contributions: dict[str, float]) -> dict[str, float]:
    """Just the pairwise interaction terms."""
    return {key: value for key, value in contributions.items() if "*" in key}


def dominant_signals(contributions: dict[str, float], top: int = 2) -> list[tuple[str, float]]:
    """The strongest per-signal contributions, largest first."""
    ordered = sorted(signal_contributions(contributions).items(), key=lambda kv: -kv[1])
    return ordered[:top]


def logit_of(contributions: dict[str, float]) -> float:
    """The log-odds the contributions add up to."""
    return float(sum(contributions.values()))


def model_status() -> dict[str, Any]:
    """What is actually running, for the report's methodology footnote."""
    return {
        "fusion_model": "ExplainableBoostingClassifier" if _load_model() is not None else "heuristic weighted logistic",
        "model_path": MODEL_PATH or "(empty slot)",
        "calibrator": "fitted" if (CALIBRATOR_PATH and os.path.exists(CALIBRATOR_PATH)) else "identity (none fitted)",
        "interpret_available": HAS_INTERPRET,
        "signal_order": list(SIGNAL_ORDER),
        "intercept": INTERCEPT,
    }
