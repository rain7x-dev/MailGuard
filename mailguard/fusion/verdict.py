"""Mapping a fused probability to a verdict class and an action.

THREE ACTIONS, NOT TWO, AND THE ARITHMETIC THAT FORCES IT

Take a mail flow of 100,000 messages a day, which is a mid-sized
enterprise. A detector running at a 0.1 percent false positive rate is
doing well by any published benchmark. 0.1 percent of 100,000 is 100
legitimate messages affected every day. That is 100 invoices, customer
replies and job applications a day, roughly 3,000 a month.

A system that hard blocks all 100 of them is switched off within a
fortnight, and then it catches nothing at all. The middle band exists for
exactly those messages: anything the model is not confident about is
marked WARN and still reaches the user, carrying a visible banner and an
analyst release path. The user keeps their mail, the analyst gets a queue
ordered by probability, and the operator keeps the tool.

So:

    PASS   deliver normally
    WARN   deliver with a banner, queue for an analyst
    BLOCK  quarantine, release only by an analyst

The cut points below are not arbitrary and the reasoning for each one is
in THRESHOLDS. Five verdict classes give the analyst the shape of the
attack, not just its strength: a message can be strongly suspicious
because of who it claims to be from (Impersonated) or because of what it
asks for (Phishing), and those two go to different playbooks.
"""
from __future__ import annotations

from typing import Iterable, Optional

if __package__ in (None, ""):  # pragma: no cover - script convenience only
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from mailguard.core.models import SignalResult

# ----------------------------------------------------------------------
# Cut points. Every number here is a policy decision, so every number
# gets a reason.
# ----------------------------------------------------------------------
THRESHOLDS: dict[str, float] = {
    # Below this the mail is delivered untouched. Set at 0.35 rather than
    # 0.5 because the cost of a missed fraud is asymmetric with the cost
    # of a banner: a banner costs a second of attention, a successful
    # wire fraud costs the quarter. The band from 0.35 to 0.5 is where
    # most first-contact-plus-urgency mail lands, and it is worth a
    # banner.
    "pass_below": 0.35,
    # At or above this the mail is quarantined. 0.80 is chosen against the
    # fusion weights so that BLOCK requires corroboration: two
    # independent strong findings, or one strong finding together with
    # its interaction term. No single signal at the top of its range
    # reaches it alone, because a lone maxed out signal is
    # indistinguishable from a broken signal, and because that is what
    # keeps the daily block volume inside what an analyst can review.
    # So a lookalike domain with nothing else wrong reaches WARN at most.
    # In practice it does reach WARN, because a newly registered domain
    # also has no sender baseline, so x4 abstains and its missingness term
    # is added on top. Either way the mail still reaches the user and the
    # analyst still sees it.
    "block_at": 0.80,
    # Class boundary only, no action attached: above this the mail is
    # named by attack type rather than called merely suspicious.
    "named_attack_at": 0.60,
    # Identity and intent scores at or above this each count as "high"
    # when deciding between Impersonated, Phishing and Fraud.
    "dominant_signal_high": 0.60,
}

VERDICT_CLASSES: tuple[str, ...] = (
    "Legitimate",
    "Suspicious",
    "Impersonated",
    "Phishing",
    "Fraud",
)

ACTIONS: tuple[str, ...] = ("BLOCK", "WARN", "PASS")

# Which signal carries which kind of attack. Kept as a constant so the
# mapping is data an operator can read, not logic buried in a function.
IDENTITY_SIGNAL: str = "x2"
INTENT_SIGNAL: str = "x5"

# Statuses that mean the score cannot be used. An abstaining signal must
# never be read as a low score, here or anywhere else.
UNUSABLE_STATUSES: frozenset[str] = frozenset({"abstain", "error", "skipped", "unavailable"})


def _usable_score(signals: Iterable[SignalResult], signal_id: str) -> Optional[float]:
    """Score of one signal, or None if it is absent or abstaining."""
    for signal in signals or []:
        if signal.signal_id != signal_id:
            continue
        if (signal.status or "").lower() in UNUSABLE_STATUSES:
            return None
        return float(signal.score)
    return None


def to_action(prob: float) -> str:
    """Map a calibrated probability to BLOCK, WARN or PASS."""
    probability = max(0.0, min(1.0, float(prob)))
    if probability >= THRESHOLDS["block_at"]:
        return "BLOCK"
    if probability < THRESHOLDS["pass_below"]:
        return "PASS"
    return "WARN"


def to_verdict_class(prob: float, signals: Iterable[SignalResult]) -> str:
    """Map a probability plus the signal set to one of five classes.

    Below pass_below the mail is Legitimate; between there and
    named_attack_at it is Suspicious, which is an honest label for "this
    is off but we cannot say what it is".

    Above named_attack_at the class names the attack, chosen by which of
    the two carrying signals dominates:

        identity high and intent high   -> Fraud
        identity higher than intent     -> Impersonated
        intent higher than identity     -> Phishing

    Neither one high above the threshold means the probability was built
    out of the other signals (a payload, an origin, a broken baseline).
    That is a real finding but not a named attack type, so it stays
    Suspicious below block_at and is called Fraud at or above it, where
    the evidence table has to carry the naming instead.
    """
    probability = max(0.0, min(1.0, float(prob)))
    signal_list = list(signals or [])

    if probability < THRESHOLDS["pass_below"]:
        return "Legitimate"
    if probability < THRESHOLDS["named_attack_at"]:
        return "Suspicious"

    identity = _usable_score(signal_list, IDENTITY_SIGNAL)
    intent = _usable_score(signal_list, INTENT_SIGNAL)
    high = THRESHOLDS["dominant_signal_high"]

    identity_value = identity if identity is not None else 0.0
    intent_value = intent if intent is not None else 0.0
    identity_high = identity is not None and identity_value >= high
    intent_high = intent is not None and intent_value >= high

    if identity_high and intent_high:
        # A forged identity making a fraudulent request is the complete
        # attack, not two separate oddities.
        return "Fraud"
    if identity_high and not intent_high:
        return "Impersonated"
    if intent_high and not identity_high:
        return "Phishing"

    # Neither carrier is high. Fall back to whichever is larger once the
    # probability is high enough to have to name something.
    if probability >= THRESHOLDS["block_at"]:
        if identity_value > intent_value:
            return "Impersonated"
        if intent_value > identity_value:
            return "Phishing"
        return "Fraud"
    return "Suspicious"


def describe_action(action: str) -> str:
    """One line of operator guidance for each action."""
    return {
        "BLOCK": "Quarantined. Release only after analyst review.",
        "WARN": "Delivered with a warning banner and queued for analyst review.",
        "PASS": "Delivered normally.",
    }.get(action, "Unknown action.")


def band_summary() -> list[tuple[str, str, str]]:
    """The three bands as printable rows: (action, range, reasoning)."""
    return [
        (
            "PASS",
            f"p < {THRESHOLDS['pass_below']:.2f}",
            "delivered normally",
        ),
        (
            "WARN",
            f"{THRESHOLDS['pass_below']:.2f} <= p < {THRESHOLDS['block_at']:.2f}",
            "delivered with a banner, queued for an analyst, releasable by the user",
        ),
        (
            "BLOCK",
            f"p >= {THRESHOLDS['block_at']:.2f}",
            "quarantined, analyst release only",
        ),
    ]
