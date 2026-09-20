"""Tests for the fusion layer, the verdict mapping and the ledger.

Plain asserts, no test framework required:

    python tests/test_fusion.py

pytest will also collect and run these, because every check lives in a
`test_*` function, but nothing here depends on pytest being installed.

The test that matters most is test_abstain_is_not_zero: the distinction
between "this signal could not run" and "this signal ran and found
nothing" is the behaviour most likely to be broken by a well meaning
refactor, and breaking it silently rewards exactly the sender profile an
attacker presents.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mailguard.core.models import ParsedEmail, SignalResult
from mailguard.evidence.ledger import EvidenceLedger, case_id_for
from mailguard.fusion.ebm_fusion import (
    MISSING_WEIGHTS,
    SIGNAL_ORDER,
    build_feature_vector,
    fuse,
    logit_of,
    signal_contributions,
)
from mailguard.fusion.verdict import THRESHOLDS, to_action, to_verdict_class
from tests.factory import make_email, make_signals, make_verdict


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def signal(signal_id: str, score: float, status: str = "ok") -> SignalResult:
    return SignalResult(
        signal_id=signal_id,
        name=signal_id.upper(),
        score=score,
        status=status,
        evidence_row=f"{signal_id} test row",
    )


def bare_email() -> ParsedEmail:
    """A minimal ParsedEmail: fusion must not need a populated message."""
    return ParsedEmail(
        message_id="<test@example.invalid>",
        raw_sha256="0" * 64,
        raw_bytes=b"",
        headers={},
        from_display="",
        from_address="test@example.invalid",
        from_domain="example.invalid",
        to_addresses=["victim@example.invalid"],
        subject="",
        text_body="",
        html_body="",
        urls=[],
        attachments=[],
        inline_images=[],
        dkim_signatures=[],
        received_chain=[],
        references=[],
    )


def seven(scores: dict[str, float], statuses: dict[str, str] | None = None) -> list[SignalResult]:
    statuses = statuses or {}
    return [
        signal(sid, scores.get(sid, 0.0), statuses.get(sid, "ok")) for sid in SIGNAL_ORDER
    ]


# ----------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------
def test_seven_ok_signals_produce_a_probability() -> None:
    """Fusing seven usable signals gives a probability strictly in (0, 1)."""
    email = bare_email()
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        verdict = fuse(seven({sid: level for sid in SIGNAL_ORDER}), email)
        assert 0.0 < verdict.probability < 1.0, f"probability out of range at {level}"
        assert verdict.action in ("BLOCK", "WARN", "PASS")
        assert verdict.verdict_class in (
            "Legitimate", "Suspicious", "Impersonated", "Phishing", "Fraud",
        )
    # Monotone in the signal scores, which any sane fusion must be.
    low = fuse(seven({sid: 0.1 for sid in SIGNAL_ORDER}), email).probability
    high = fuse(seven({sid: 0.9 for sid in SIGNAL_ORDER}), email).probability
    assert low < high, "raising every signal must not lower the probability"


def test_feature_vector_is_fourteen_values_with_missingness() -> None:
    """Seven scores plus seven presence flags, in SIGNAL_ORDER."""
    signals = seven({"x2": 0.8, "x5": 0.4}, statuses={"x4": "abstain"})
    signals = [s for s in signals if s.signal_id != "x6"]  # drop one entirely
    vector, by_id = build_feature_vector(signals)
    assert len(vector) == 14, "vector must carry seven scores and seven flags"

    scores, presence = vector[:7], vector[7:]
    index = {sid: position for position, sid in enumerate(SIGNAL_ORDER)}
    assert presence[index["x4"]] == 0.0, "an abstaining signal is not present"
    assert presence[index["x6"]] == 0.0, "a missing signal is not present"
    assert presence[index["x2"]] == 1.0
    assert scores[index["x2"]] == 0.8
    assert scores[index["x4"]] == 0.0, "an unusable score is zeroed, the flag carries the meaning"
    assert by_id["x6"] is None


def test_abstain_is_not_zero() -> None:
    """An abstaining signal must NOT be treated the same as a score of 0.0.

    This is the single most important behaviour in the fusion layer. A
    brand new sender with no baseline is exactly the profile an attacker
    presents, so scoring "could not measure" as a clean neutral would
    reward the attack.
    """
    email = bare_email()
    base = {"x1": 0.0, "x2": 0.4, "x3": 0.0, "x5": 0.3, "x6": 0.0, "x7": 0.0}

    abstained = fuse(seven({**base, "x4": 0.0}, statuses={"x4": "abstain"}), email)
    scored_zero = fuse(seven({**base, "x4": 0.0}), email)

    assert abstained.probability != scored_zero.probability, (
        "abstain and a 0.0 score produced the same probability: missing data is "
        "being read as evidence of innocence"
    )
    assert abstained.probability > scored_zero.probability, (
        "an unmeasurable signal must not be more reassuring than a measured clean one"
    )
    assert abstained.contributions["x4"] == MISSING_WEIGHTS["x4"], (
        "an absent signal should contribute its missingness prior"
    )
    assert scored_zero.contributions["x4"] == 0.0, (
        "a measured clean signal should contribute nothing"
    )

    # A signal missing from the list entirely behaves like an abstention.
    dropped = fuse([s for s in seven(base) if s.signal_id != "x4"], email)
    assert abs(dropped.probability - abstained.probability) < 1e-9, (
        "a signal that never ran and one that abstained should score the same"
    )


def test_contributions_sum_to_the_logit() -> None:
    """The printed evidence table must add up to the verdict."""
    import math

    email = bare_email()
    verdict = fuse(seven({"x1": 0.1, "x2": 0.7, "x3": 0.2, "x5": 0.6, "x7": 0.5},
                         statuses={"x4": "abstain"}), email)
    total = logit_of(verdict.contributions)
    expected = math.log(verdict.probability / (1.0 - verdict.probability))
    assert abs(total - expected) < 1e-3, (
        f"contributions sum to {total:.4f} but the probability implies {expected:.4f}"
    )
    assert "_intercept" in verdict.contributions, "the base rate must be itemised"
    assert set(signal_contributions(verdict.contributions)) == set(SIGNAL_ORDER), (
        "every signal needs a contribution, including the ones that abstained"
    )


def test_authenticated_bec_still_blocks() -> None:
    """The flagship case: x1 passes, x2 and x5 are high, the mail is BLOCKed.

    The attacker owns their cousin domain and publishes correct SPF and
    DKIM for it, so authentication is clean. The verdict does not depend
    on authentication, which is the entire argument for scoring identity
    and intent separately.
    """
    email = bare_email()
    signals = seven(
        {"x1": 0.0, "x2": 0.9, "x3": 0.0, "x4": 0.0, "x5": 0.85, "x6": 0.0, "x7": 0.0}
    )
    verdict = fuse(signals, email)
    assert verdict.action == "BLOCK", (
        f"authenticated BEC was not blocked: p={verdict.probability:.4f}"
    )
    assert verdict.probability >= THRESHOLDS["block_at"]
    assert verdict.verdict_class == "Fraud", (
        "identity and intent both high is the complete attack, not one of its halves"
    )
    # x1 contributed nothing, and the verdict stands anyway.
    assert verdict.contributions["x1"] == 0.0
    assert "x2*x5" in verdict.contributions, "the identity/intent interaction must be itemised"


def test_full_pipeline_on_the_factory_mail() -> None:
    """The realistic fixture end to end: BLOCK, Fraud, abstention intact."""
    fixture_signals = make_signals()
    assert len(fixture_signals) == 7, "the fixture must cover all seven signals"
    assert sum(1 for s in fixture_signals if s.status == "abstain") >= 1
    verdict = make_verdict(make_email(), fixture_signals)
    assert verdict.action == "BLOCK"
    assert verdict.verdict_class == "Fraud"
    assert 0.0 < verdict.probability < 1.0
    abstaining = [s for s in verdict.signals if s.status == "abstain"]
    assert abstaining, "the fixture must exercise the abstain path"
    assert all(s.signal_id in verdict.contributions for s in verdict.signals)


# ----------------------------------------------------------------------
# Verdict classes and actions
# ----------------------------------------------------------------------
def test_actions_follow_the_three_bands() -> None:
    assert to_action(0.0) == "PASS"
    assert to_action(THRESHOLDS["pass_below"] - 0.01) == "PASS"
    assert to_action(THRESHOLDS["pass_below"]) == "WARN"
    assert to_action(THRESHOLDS["block_at"] - 0.01) == "WARN"
    assert to_action(THRESHOLDS["block_at"]) == "BLOCK"
    assert to_action(1.0) == "BLOCK"


def test_verdict_class_identity_versus_intent() -> None:
    """Identity dominant gives Impersonated, intent dominant gives Phishing."""
    identity_heavy = seven({"x2": 0.92, "x5": 0.10})
    intent_heavy = seven({"x2": 0.10, "x5": 0.92})
    both_heavy = seven({"x2": 0.88, "x5": 0.85})

    assert to_verdict_class(0.90, identity_heavy) == "Impersonated"
    assert to_verdict_class(0.90, intent_heavy) == "Phishing"
    assert to_verdict_class(0.90, both_heavy) == "Fraud"

    # Below the naming threshold the class stays honest about uncertainty.
    assert to_verdict_class(0.10, identity_heavy) == "Legitimate"
    assert to_verdict_class(0.45, identity_heavy) == "Suspicious"


def test_verdict_class_end_to_end() -> None:
    """The same distinction, through fuse() rather than the mapper alone."""
    email = bare_email()
    impersonation = fuse(
        seven({"x1": 0.05, "x2": 0.95, "x3": 0.05, "x5": 0.10, "x6": 0.05, "x7": 0.5},
              statuses={"x4": "abstain"}),
        email,
    )
    phishing = fuse(
        seven({"x1": 0.05, "x2": 0.10, "x3": 0.05, "x5": 0.95, "x6": 0.05, "x7": 0.5},
              statuses={"x4": "abstain"}),
        email,
    )
    assert impersonation.verdict_class == "Impersonated", impersonation.verdict_class
    assert phishing.verdict_class == "Phishing", phishing.verdict_class


def test_abstaining_identity_is_not_read_as_a_low_score() -> None:
    """An abstaining x2 must not be used to argue the mail is not impersonation."""
    signals = seven({"x5": 0.95}, statuses={"x2": "abstain"})
    assert to_verdict_class(0.90, signals) == "Phishing"
    # And with both carriers abstaining, no attack type may be asserted from them.
    both_out = seven({}, statuses={"x2": "abstain", "x5": "abstain"})
    assert to_verdict_class(0.90, both_out) == "Fraud"
    assert to_verdict_class(0.65, both_out) == "Suspicious"


# ----------------------------------------------------------------------
# Ledger
# ----------------------------------------------------------------------
def test_ledger_detects_a_tampered_entry() -> None:
    """Editing an entry breaks the chain, and the break is reported by index."""
    import json

    directory = tempfile.mkdtemp(prefix="mailguard-test-ledger-")
    ledger = EvidenceLedger(os.path.join(directory, "chain.jsonl"))
    email = make_email()
    case = case_id_for(email.raw_sha256)

    ledger.seal_raw(email.raw_sha256, source="smtp://test", operator="test.runner")
    ledger.append({"type": "parsed", "case_id": case, "hops": 4})
    ledger.append({"type": "signals", "case_id": case, "x2": 0.91})
    ledger.append({"type": "verdict", "case_id": case, "action": "BLOCK"})

    intact, broken_at = ledger.verify_chain()
    assert intact is True and broken_at is None, "a fresh chain must verify"
    assert ledger.count() == 4

    # Tamper with entry 2 and re-verify.
    path = ledger.path
    lines = open(path, "r", encoding="utf-8").read().splitlines()
    entry = json.loads(lines[2])
    entry["payload"]["x2"] = 0.01
    lines[2] = json.dumps(entry, sort_keys=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    intact, broken_at = ledger.verify_chain()
    assert intact is False, "an edited entry must not verify"
    assert broken_at == 2, f"expected the break at index 2, got {broken_at}"


def test_seal_must_come_first() -> None:
    """seal_raw refuses to run after other entries exist for the same case."""
    directory = tempfile.mkdtemp(prefix="mailguard-test-order-")
    ledger = EvidenceLedger(os.path.join(directory, "chain.jsonl"))
    email = make_email()
    case = case_id_for(email.raw_sha256)
    ledger.append({"type": "parsed", "case_id": case, "hops": 4})
    try:
        ledger.seal_raw(email.raw_sha256, source="smtp://test", operator="test.runner")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "sealing after processing was allowed; the ordering guarantee is gone"
        )


# ----------------------------------------------------------------------
# Signals owned by this half, smoke tested through the real fixture
# ----------------------------------------------------------------------
def test_owned_signals_run_and_never_raise() -> None:
    """x2, x4, x5 and x7 all return a SignalResult for the fixture mail."""
    from mailguard.signals import x2_identity, x4_sender_baseline, x5_intent, x7_attachment_url

    email = make_email()
    for module in (x2_identity, x4_sender_baseline, x5_intent, x7_attachment_url):
        result = module.run(email)
        assert isinstance(result, SignalResult)
        assert result.signal_id == module.SIGNAL_ID
        assert 0.0 <= result.score <= 1.0
        assert result.evidence_row, f"{module.SIGNAL_ID} produced no evidence row"

    # The fixture is a cousin domain attack with a macro payload, so
    # identity, intent and payload should all fire; the sender has no
    # history, so the baseline must abstain rather than score zero.
    assert x2_identity.run(email).score >= 0.5
    assert x5_intent.run(email).score >= 0.5
    assert x7_attachment_url.run(email).score >= 0.5
    baseline = x4_sender_baseline.run(email)
    assert baseline.status == "abstain", "a sender with no history has no baseline"
    assert baseline.score == 0.0


def test_signals_never_raise_on_garbage() -> None:
    """A malformed message must produce abstentions, not exceptions."""
    from mailguard.signals import x2_identity, x4_sender_baseline, x5_intent, x7_attachment_url

    broken = bare_email()
    broken.from_address = ""
    broken.from_domain = ""
    broken.received_chain = []
    for module in (x2_identity, x4_sender_baseline, x5_intent, x7_attachment_url):
        result = module.run(broken)
        assert isinstance(result, SignalResult)
        assert result.status in ("ok", "abstain")


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def _all_tests() -> list[tuple[str, Callable[[], None]]]:
    module = sys.modules[__name__]
    return [
        (name, getattr(module, name))
        for name in sorted(dir(module))
        if name.startswith("test_") and callable(getattr(module, name))
    ]


def main() -> int:
    passed, failed = 0, 0
    print("MailGuard fusion, verdict and ledger tests")
    print("=" * 78)
    for name, test in _all_tests():
        try:
            test()
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            print(
                "      "
                + "\n      ".join(traceback.format_exc().strip().splitlines()[-6:])
            )
        else:
            passed += 1
            print(f"ok    {name}")
    print("-" * 78)
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
