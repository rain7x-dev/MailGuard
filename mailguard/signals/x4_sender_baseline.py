"""x4 Sender Baseline: does this message look like this sender?

Every other signal in MailGuard looks for something wrong with the mail.
This one looks for something wrong with the *sender*, by comparing the
message against what that address has done before. It is the only signal
in the set that can catch a genuinely compromised legitimate account,
because in that case every other signal is clean by definition: the
domain is real, SPF and DKIM pass, the display name is correct, the
thread is real, the account is old and trusted. The only thing that
changed is the person at the keyboard.

Three comparisons:

1. Embedding drift. How far this body sits from the mean of that
   sender's previous bodies. Backed by sentence-transformers when it is
   installed, TF-IDF cosine distance when scikit-learn is, and a pure
   Python character n-gram Jaccard distance when neither is. All three
   answer the same question with decreasing sharpness, so the signal
   works on a laptop with no wheels installed.
2. Send hour deviation. A finance clerk who has written at 09:00-18:00
   IST for two years and suddenly writes at 03:40 is worth a look.
   Compared with circular statistics, because hours wrap at midnight and
   a linear mean would put the average of 23:00 and 01:00 at noon.
3. Volume anomaly. Compromised accounts get used hard and briefly.

THE ABSTAIN RULE, which is the most important behaviour in this file:
if the sender has fewer than MIN_HISTORY_MESSAGES prior messages there
is no baseline, and this signal returns status="abstain" with score 0.0.
A missing baseline is NOT evidence of innocence. A brand new sender with
no history is exactly the profile an attacker presents, so scoring it as
a clean 0.0 would actively reward the attack. Fusion carries a
missingness indicator per signal precisely so that "we could not measure
this" stays distinguishable from "we measured this and it was fine".
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Optional

if __package__ in (None, ""):  # pragma: no cover - script convenience only
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from mailguard.core.models import ParsedEmail, SignalResult

SIGNAL_ID = "x4"
SIGNAL_NAME = "Sender Baseline"

# ----------------------------------------------------------------------
# Optional dependencies. Each is guarded, each has a documented fallback,
# and none of them is imported or loaded at module import time.
# ----------------------------------------------------------------------
try:
    import sentence_transformers  # noqa: F401  (presence check only)

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import sklearn  # noqa: F401  (presence check only)

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ======================================================================
# TRAINED MODEL SLOT
# ----------------------------------------------------------------------
# The model for this signal has NOT been trained yet.
# When it is, put the artefact path in MODEL_PATH below and the code
# will pick it up automatically on the next run. Until then the
# documented heuristic in _heuristic_score() runs instead, so the whole
# pipeline stays runnable end to end today.
#
# Expected artefact: a joblib dumped dict
#     {
#       "model": <fitted sklearn.ensemble.IsolationForest>,
#       "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
#       "feature_names": [...],          # must equal FEATURE_NAMES
#       "score_scale": [lo, hi]          # optional min/max of
#                                        # -score_samples() on the
#                                        # training set, for mapping the
#                                        # anomaly score into 0..1
#     }
# Fit the forest on per-sender feature rows drawn from known-good mail
# only: this is an outlier detector, not a classifier, because there is
# no useful corpus of "this specific account, compromised". The
# embedding model reference must match the one used at training time or
# the drift feature means something different at inference.
# Expected feature order is defined by FEATURE_NAMES below.
# ======================================================================
MODEL_PATH: str = ""          # <-- trained model path goes here

FEATURE_NAMES: list[str] = [
    "embedding_drift",          # 0..1 distance from the sender's mean body
    "send_hour_deviation",      # 0..1 how far outside the normal window
    "volume_anomaly",           # 0..1 burst relative to the normal rate
    "history_count_capped",     # prior messages seen, capped at 200
    "days_since_last_contact",  # dormant accounts get reused
    "new_recipient",            # this sender has not written to this To before
    "body_length_ratio",        # |len(body) / mean(len(history))| folded to 0..1
    "off_hours_send",           # sent between 00:00 and 05:00 local to the header
]

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Below this many prior messages there is no baseline and the signal
# abstains. Five is low on purpose: it is enough to establish an hour
# window and a rough voice, and holding out for thirty would mean
# abstaining on most real corporate correspondents.
MIN_HISTORY_MESSAGES: int = int(os.environ.get("MAILGUARD_MIN_BASELINE", "5"))

# Same JSON store x2 reads. See the shape documented in x2_identity.py.
# Duplicated reader, shared file: in production both signals call one
# SenderHistoryRepository instead, but each signal module has to stay
# independently loadable by the signal loader.
HISTORY_PATH: str = os.environ.get("MAILGUARD_HISTORY_PATH", "")

# Embedding model used when sentence-transformers is available. Small,
# CPU friendly, and the same default the training slot documents.
EMBEDDING_MODEL_NAME: str = os.environ.get(
    "MAILGUARD_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

_MODEL: Any = None
_MODEL_TRIED: bool = False
_EMBEDDER: Any = None
_EMBEDDER_TRIED: bool = False
_HISTORY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _load_history(path: Optional[str] = None) -> dict[str, Any]:
    """Read the JSON sender history store, or return an empty one."""
    store_path = HISTORY_PATH if path is None else path
    empty: dict[str, Any] = {"senders": {}, "message_ids": []}
    if not store_path or not os.path.exists(store_path):
        return empty
    try:
        mtime = os.path.getmtime(store_path)
        cached = _HISTORY_CACHE.get(store_path)
        if cached and cached[0] == mtime:
            return cached[1]
        with open(store_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return empty
        data.setdefault("senders", {})
        data.setdefault("message_ids", [])
        _HISTORY_CACHE[store_path] = (mtime, data)
        return data
    except (OSError, ValueError):
        return empty


# ----------------------------------------------------------------------
# Embedding drift, three backends
# ----------------------------------------------------------------------
def _get_embedder() -> Any:
    """Load the sentence-transformers model once, lazily, or return None.

    Deliberately not at import time: loading a transformer costs seconds
    and hundreds of megabytes, and a module that does that on import
    cannot be part of a CLI that also has to run `--help`.
    """
    global _EMBEDDER, _EMBEDDER_TRIED
    if _EMBEDDER_TRIED:
        return _EMBEDDER
    _EMBEDDER_TRIED = True
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception:
        _EMBEDDER = None
    return _EMBEDDER


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    similarity = dot / (na * nb)
    # Cosine similarity of text embeddings lives in roughly [-1, 1]; fold
    # it to a 0..1 distance and clamp, because a negative similarity is
    # already maximally different for our purposes.
    return max(0.0, min(1.0, (1.0 - similarity)))


def _drift_sentence_transformers(current: str, history: list[str]) -> Optional[float]:
    """Tier 1: semantic drift from a sentence embedding model."""
    embedder = _get_embedder()
    if embedder is None:
        return None
    try:
        vectors = embedder.encode([current] + history, convert_to_numpy=False)
        vectors = [list(map(float, v)) for v in vectors]
        current_vec, history_vecs = vectors[0], vectors[1:]
        if not history_vecs:
            return None
        mean_vec = [sum(col) / len(history_vecs) for col in zip(*history_vecs)]
        return _cosine_distance(current_vec, mean_vec)
    except Exception:
        return None


def _drift_tfidf(current: str, history: list[str]) -> Optional[float]:
    """Tier 2: lexical drift from TF-IDF cosine distance."""
    if not HAS_SKLEARN:
        return None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [current] + history
        matrix = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        ).fit_transform(corpus)
        dense = matrix.toarray()
        current_vec = [float(x) for x in dense[0]]
        history_rows = dense[1:]
        if len(history_rows) == 0:
            return None
        mean_vec = [float(sum(col) / len(history_rows)) for col in zip(*history_rows)]
        return _cosine_distance(current_vec, mean_vec)
    except Exception:
        return None


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    cleaned = " ".join((text or "").lower().split())
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def _drift_jaccard(current: str, history: list[str]) -> float:
    """Tier 3: character 4-gram Jaccard distance, pure Python.

    Blunter than an embedding: it sees spelling, punctuation habits and
    boilerplate rather than meaning. That is still useful, because a
    compromised account usually loses the sender's signature block and
    quoting style before it loses their vocabulary.
    """
    current_grams = _char_ngrams(current)
    if not current_grams:
        return 0.0
    distances: list[float] = []
    for past in history:
        past_grams = _char_ngrams(past)
        if not past_grams:
            continue
        union = current_grams | past_grams
        if not union:
            continue
        distances.append(1.0 - len(current_grams & past_grams) / len(union))
    if not distances:
        return 0.0
    # Nearest prior message, not the mean: a sender with two distinct
    # modes (short replies and long reports) should not read as drifting
    # just because this mail matches only one of them.
    return max(0.0, min(1.0, min(distances)))


def embedding_drift(current: str, history: list[str]) -> tuple[float, str]:
    """Distance between this body and the sender's previous bodies.

    Returns (drift in 0..1, name of the backend that produced it) so the
    report can state which method was used. Evidence that does not say
    how it was measured is not evidence.
    """
    history = [h for h in (history or []) if (h or "").strip()]
    if not (current or "").strip() or not history:
        return 0.0, "unavailable"
    value = _drift_sentence_transformers(current, history)
    if value is not None:
        return value, "sentence-transformers"
    value = _drift_tfidf(current, history)
    if value is not None:
        return value, "tfidf-cosine"
    return _drift_jaccard(current, history), "char-ngram-jaccard"


# ----------------------------------------------------------------------
# Time and volume statistics
# ----------------------------------------------------------------------
def _circular_hour_stats(hours: list[float]) -> tuple[float, float]:
    """Circular mean and circular standard deviation of clock hours.

    Hours wrap at midnight, so a linear mean of 23:00 and 01:00 would be
    noon, which is exactly wrong for a signal about unusual send times.
    """
    if not hours:
        return 0.0, 12.0
    angles = [2.0 * math.pi * (h % 24) / 24.0 for h in hours]
    sin_sum = sum(math.sin(a) for a in angles) / len(angles)
    cos_sum = sum(math.cos(a) for a in angles) / len(angles)
    mean_angle = math.atan2(sin_sum, cos_sum) % (2.0 * math.pi)
    mean_hour = mean_angle * 24.0 / (2.0 * math.pi)
    resultant = math.sqrt(sin_sum * sin_sum + cos_sum * cos_sum)
    if resultant <= 1e-9:
        return mean_hour, 12.0
    circular_sd_rad = math.sqrt(-2.0 * math.log(min(resultant, 1.0)))
    return mean_hour, circular_sd_rad * 24.0 / (2.0 * math.pi)


def _hour_distance(a: float, b: float) -> float:
    """Shortest distance between two clock hours, 0..12."""
    diff = abs((a % 24) - (b % 24))
    return min(diff, 24.0 - diff)


def send_hour_deviation(hour: Optional[float], history_hours: list[float]) -> tuple[float, dict[str, float]]:
    """How far outside this sender's normal window this mail was sent.

    Inside one circular standard deviation scores 0. Four deviations out
    scores 1. The tolerance floor of one hour stops a sender with a
    metronomic schedule from flagging on a twenty minute slip.
    """
    context = {"mean_hour": 0.0, "spread_hours": 0.0, "distance_hours": 0.0}
    if hour is None or not history_hours:
        return 0.0, context
    mean_hour, spread = _circular_hour_stats(history_hours)
    spread = max(spread, 1.0)
    distance = _hour_distance(hour, mean_hour)
    context.update(mean_hour=mean_hour, spread_hours=spread, distance_hours=distance)
    z = distance / spread
    return max(0.0, min(1.0, (z - 1.0) / 3.0)), context


def volume_anomaly(today_count: float, daily_counts: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Today's message count against this sender's normal daily rate."""
    context = {"today": float(today_count), "mean_per_day": 0.0, "sd_per_day": 0.0}
    values: list[float] = []
    for value in (daily_counts or {}).values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values or today_count <= 0:
        return 0.0, context
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    sd = math.sqrt(variance)
    context.update(mean_per_day=mean, sd_per_day=sd)
    if mean <= 0:
        return 0.0, context
    # Five times the normal rate is a burst whatever the variance says.
    if today_count >= 5.0 * mean:
        return 1.0, context
    tolerance = max(sd, 1.0)
    return max(0.0, min(1.0, (today_count - mean) / (3.0 * tolerance))), context


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def _bare_address(value: Optional[str]) -> str:
    if not value:
        return ""
    value = value.strip()
    if "<" in value and ">" in value:
        value = value[value.rfind("<") + 1 : value.rfind(">")]
    return value.strip().lower()


def _extract_details(email: ParsedEmail) -> dict[str, Any]:
    """Features plus the analyst context, including the abstain reason."""
    sender = _bare_address(email.from_address)
    history = _load_history()
    record: dict[str, Any] = (history.get("senders", {}) or {}).get(sender) or {}

    try:
        history_count = int(float(record.get("count", 0) or 0))
    except (TypeError, ValueError):
        history_count = 0

    bodies = [str(b) for b in (record.get("bodies") or []) if str(b).strip()]
    hours_raw = record.get("hours") or []
    history_hours: list[float] = []
    for value in hours_raw:
        try:
            history_hours.append(float(value))
        except (TypeError, ValueError):
            continue

    body = (email.text_body or "").strip() or (email.html_body or "").strip()
    drift, backend = embedding_drift(body, bodies)

    hour = float(email.date.hour) + float(email.date.minute) / 60.0 if email.date else None
    hour_dev, hour_context = send_hour_deviation(hour, history_hours)

    daily_counts: dict[str, Any] = record.get("daily_counts") or {}
    day_key = email.date.date().isoformat() if email.date else ""
    try:
        today_count = float(daily_counts.get(day_key, 0) or 0)
    except (TypeError, ValueError):
        today_count = 0.0
    # The message in hand counts too, otherwise a first burst message
    # always reads as zero volume.
    volume, volume_context = volume_anomaly(today_count + 1.0, daily_counts)

    known_recipients = {str(r).lower() for r in (record.get("recipients") or [])}
    recipients = {str(r).lower() for r in (email.to_addresses or [])}
    new_recipient = (
        1.0 if known_recipients and recipients and not (recipients & known_recipients) else 0.0
    )

    mean_body_len = (sum(len(b) for b in bodies) / len(bodies)) if bodies else 0.0
    if mean_body_len > 0 and body:
        ratio = len(body) / mean_body_len
        # Fold the ratio so that "half as long" and "twice as long" score
        # the same: both are a change of habit.
        length_ratio = min(1.0, abs(math.log(max(ratio, 1e-3))) / math.log(4.0))
    else:
        length_ratio = 0.0

    days_since = 0.0
    last_seen = record.get("last_seen")
    if last_seen and email.date:
        try:
            parsed_last = datetime.fromisoformat(str(last_seen))
            if parsed_last.tzinfo is not None and email.date.tzinfo is None:
                parsed_last = parsed_last.replace(tzinfo=None)
            delta_days = (email.date - parsed_last).days
            # Normalised against a quarter: a dormant account coming back
            # after ninety days is the interesting end of the scale.
            days_since = max(0.0, min(1.0, delta_days / 90.0))
        except (TypeError, ValueError):
            days_since = 0.0

    off_hours = 1.0 if hour is not None and 0.0 <= hour < 5.0 else 0.0

    return {
        "embedding_drift": drift,
        "send_hour_deviation": hour_dev,
        "volume_anomaly": volume,
        "history_count_capped": float(min(history_count, 200)),
        "days_since_last_contact": days_since,
        "new_recipient": new_recipient,
        "body_length_ratio": length_ratio,
        "off_hours_send": off_hours,
        "_sender": sender,
        "_history_count": history_count,
        "_bodies_available": len(bodies),
        "_hours_available": len(history_hours),
        "_drift_backend": backend,
        "_hour": hour,
        "_hour_context": hour_context,
        "_volume_context": volume_context,
        "_history_configured": bool(HISTORY_PATH),
    }


def _extract_features(email: ParsedEmail) -> dict[str, float]:
    """Pull every feature in FEATURE_NAMES out of the email."""
    details = _extract_details(email)
    return {name: float(details[name]) for name in FEATURE_NAMES}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def _heuristic_score(features: dict[str, float]) -> float:
    """Documented stand in until the isolation forest is trained.

    Weighted sum of the three baseline comparisons:

        0.55 * embedding_drift
      + 0.25 * send_hour_deviation
      + 0.20 * volume_anomaly

    plus 0.10 if the mail went to a recipient this sender has never
    written to, 0.05 for an off-hours send, and 0.05 for a dormant
    account waking up. Clamped to [0, 1].

    Drift dominates because it is the hardest of the three for an
    attacker to control: whoever is typing has to sound like the account
    holder, having only read the mailbox. Hour and volume are cheap to
    imitate once noticed, so they get less weight; they earn their place
    by being the two features that fire earliest in a mass-mailing
    compromise, before the attacker has settled into a style.

    Whoever trains the isolation forest should beat this on recall of
    confirmed account-takeover incidents at an equal alert budget, and
    should specifically fix the weakness here: this formula treats the
    three comparisons as independent when drift and length are plainly
    correlated.
    """
    score = (
        0.55 * features.get("embedding_drift", 0.0)
        + 0.25 * features.get("send_hour_deviation", 0.0)
        + 0.20 * features.get("volume_anomaly", 0.0)
    )
    score += 0.10 * features.get("new_recipient", 0.0)
    score += 0.05 * features.get("off_hours_send", 0.0)
    score += 0.05 * features.get("days_since_last_contact", 0.0)
    score += 0.05 * features.get("body_length_ratio", 0.0)
    return max(0.0, min(1.0, score))


def _load_model() -> Any:
    """Load the trained model once, or return None if the slot is empty."""
    global _MODEL, _MODEL_TRIED
    if _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        return None
    try:
        import joblib

        _MODEL = joblib.load(MODEL_PATH)
    except Exception:
        _MODEL = None
    return _MODEL


def _model_score(features: dict[str, float]) -> float | None:
    """Score with the trained model if the slot is filled, else None.

    An isolation forest returns a signed anomaly score, not a
    probability: score_samples() is high for inliers and low for
    outliers. We negate it and map it through the score_scale recorded at
    training time, so the number handed to fusion stays in 0..1 and keeps
    the same meaning as the heuristic it replaces.
    """
    artefact = _load_model()
    if artefact is None:
        return None
    try:
        model = artefact.get("model") if isinstance(artefact, dict) else artefact
        if model is None:
            return None
        vector = [[features.get(name, 0.0) for name in FEATURE_NAMES]]
        if hasattr(model, "score_samples"):
            raw = -float(model.score_samples(vector)[0])
        elif hasattr(model, "decision_function"):
            raw = -float(model.decision_function(vector)[0])
        else:
            return None
        scale = artefact.get("score_scale") if isinstance(artefact, dict) else None
        if scale and len(scale) == 2 and float(scale[1]) > float(scale[0]):
            lo, hi = float(scale[0]), float(scale[1])
            return max(0.0, min(1.0, (raw - lo) / (hi - lo)))
        # No recorded scale: squash with a logistic so the output is at
        # least monotone in the anomaly score and bounded.
        return 1.0 / (1.0 + math.exp(-raw))
    except Exception:
        return None


def _evidence_row(details: dict[str, Any], score: float) -> str:
    """One short printable line an analyst reads."""
    parts = [f"baseline from {details['_history_count']} prior messages"]
    drift = details["embedding_drift"]
    if details["_bodies_available"]:
        parts.append(f"body drift {drift:.2f} ({details['_drift_backend']})")
    else:
        parts.append("no prior bodies stored, drift not measured")
    hour = details["_hour"]
    hour_context = details["_hour_context"]
    if details["_hours_available"] and hour is not None:
        parts.append(
            f"sent {hour:04.1f}h against a usual {hour_context['mean_hour']:04.1f}h "
            f"+/- {hour_context['spread_hours']:.1f}h"
        )
    if details["volume_anomaly"] > 0.2:
        volume_context = details["_volume_context"]
        parts.append(
            f"{volume_context['today']:.0f} today against {volume_context['mean_per_day']:.1f} normal"
        )
    if details["new_recipient"]:
        parts.append("first mail to this recipient")
    if details["days_since_last_contact"] >= 0.5:
        parts.append("dormant account reactivated")
    row = ", ".join(parts)
    return row if len(row) <= 220 else row[:217] + "..."


def run(email: ParsedEmail) -> SignalResult:
    """Score this message against the sender's baseline. Never raises."""
    try:
        details = _extract_details(email)
        history_count = details["_history_count"]

        # ------------------------------------------------------------------
        # THE ABSTAIN RULE. A missing baseline is not evidence of
        # innocence. Returning 0.0 here would hand a clean score to every
        # brand new sender, which is precisely the profile an attacker
        # presents: a freshly registered cousin domain has no history by
        # construction. Abstaining instead tells the fusion layer "this
        # signal could not run", the missingness indicator goes up, and
        # the report prints "abstain, not scored" rather than a zero an
        # analyst would read as a pass.
        # ------------------------------------------------------------------
        if history_count < MIN_HISTORY_MESSAGES:
            reason = (
                f"no baseline: {history_count} prior messages, "
                f"{MIN_HISTORY_MESSAGES} required"
            )
            if not details["_history_configured"]:
                reason += " (no history store configured)"
            return SignalResult(
                signal_id=SIGNAL_ID,
                name=SIGNAL_NAME,
                score=0.0,
                status="abstain",
                evidence_row=reason,
                details={
                    "history_count": history_count,
                    "min_required": MIN_HISTORY_MESSAGES,
                    "history_store_configured": details["_history_configured"],
                    "abstain_reason": "insufficient sender history",
                },
            )

        # A baseline exists by count, but nothing measurable came with it.
        if not details["_bodies_available"] and not details["_hours_available"]:
            return SignalResult(
                signal_id=SIGNAL_ID,
                name=SIGNAL_NAME,
                score=0.0,
                status="abstain",
                evidence_row=(
                    f"{history_count} prior messages recorded but no bodies or send times "
                    "stored, nothing to compare against"
                ),
                details={
                    "history_count": history_count,
                    "abstain_reason": "baseline present but empty",
                },
            )

        features = {name: float(details[name]) for name in FEATURE_NAMES}
        model_value = _model_score(features)
        score = model_value if model_value is not None else _heuristic_score(features)
        public: dict[str, Any] = {k: v for k, v in details.items() if not k.startswith("_")}
        public.update(
            sender=details["_sender"],
            history_count=history_count,
            drift_backend=details["_drift_backend"],
            bodies_available=details["_bodies_available"],
            hours_available=details["_hours_available"],
            hour_context=details["_hour_context"],
            volume_context=details["_volume_context"],
            min_required=MIN_HISTORY_MESSAGES,
        )
        return SignalResult(
            signal_id=SIGNAL_ID,
            name=SIGNAL_NAME,
            score=round(float(max(0.0, min(1.0, score))), 4),
            status="ok",
            evidence_row=_evidence_row(details, score),
            details=public,
            model_backed=model_value is not None,
        )
    except Exception as exc:  # run() must never raise
        return SignalResult(
            signal_id=SIGNAL_ID,
            name=SIGNAL_NAME,
            score=0.0,
            status="abstain",
            evidence_row=f"sender baseline signal could not run: {type(exc).__name__}",
            details={"error": repr(exc)},
        )
