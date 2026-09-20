"""x5 Intent: what is this message asking the reader to do?

x2 asks whether the sender is real. This signal ignores the sender
entirely and reads the request. It carries commodity phishing, where the
content *is* the attack: a mail from a throwaway domain nobody has ever
heard of, with no lookalike trick and nothing wrong with its headers,
which simply asks the reader to log in and confirm their password.

Intent is scored across seven families, each defined as a module level
constant below so an analyst can read exactly what fired:

  * urgency framing and manufactured deadlines
  * payment redirection (pay to a different account than usual)
  * invoice manipulation (reissued, corrected, revised invoices)
  * credential harvesting (confirm your password, verify your mailbox)
  * bank detail change requests, scored separately because that single
    request is the payload of nearly every successful BEC
  * secrecy and authority pressure ("do not discuss this with anyone",
    "I am in a board meeting, handle it yourself")
  * consequence threats (suspension, legal action, penalties)

Subject line urgency is scored separately from body urgency. The subject
is where the attacker spends their attention, and a calm body under a
screaming subject is itself a pattern.

The trained model for this signal is a fine tuned DistilBERT sequence
classifier. The full inference path is written below and guarded by
ImportError, so filling in MODEL_PATH is genuinely the only remaining
step; until then the keyword and phrase scorer runs.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - script convenience only
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from mailguard.core.models import ParsedEmail, SignalResult

SIGNAL_ID = "x5"
SIGNAL_NAME = "Intent"

# ----------------------------------------------------------------------
# Optional dependencies, guarded. The heuristic below needs neither.
# ----------------------------------------------------------------------
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ======================================================================
# TRAINED MODEL SLOT
# ----------------------------------------------------------------------
# The model for this signal has NOT been trained yet.
# When it is, put the artefact path in MODEL_PATH below and the code
# will pick it up automatically on the next run. Until then the
# documented heuristic in _heuristic_score() runs instead, so the whole
# pipeline stays runnable end to end today.
#
# Expected artefact: a DIRECTORY holding a fine tuned DistilBERT
# sequence classifier, saved with save_pretrained() and loadable by
#     AutoTokenizer.from_pretrained(MODEL_PATH)
#     AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
# so it must contain config.json, the weights, and the tokenizer files.
# Two labels: benign request and fraudulent request. Put the fraud label
# in config.id2label (any of "fraud", "phish", "malicious", "1") so
# _positive_label_index() finds it rather than guessing.
# Input at inference is the subject and body joined by a newline and
# truncated to MAX_TOKENS; train it the same way or the distribution
# shifts under the model.
# Expected feature order is defined by FEATURE_NAMES below. The
# transformer consumes text rather than these features, but the features
# stay the contract for the heuristic fallback, for the evidence row, and
# for ablation when the model and the keywords disagree.
# ======================================================================
MODEL_PATH: str = ""          # <-- trained model path goes here

MAX_TOKENS: int = 512

FEATURE_NAMES: list[str] = [
    "urgency_body",              # urgency phrases in the body
    "urgency_subject",           # urgency phrases in the subject, scored apart
    "deadline_pressure",         # an explicit short deadline
    "payment_redirection",       # pay somewhere other than the usual place
    "invoice_manipulation",      # revised, corrected, reissued invoice framing
    "credential_harvest",        # asks for a password, OTP, or mailbox login
    "bank_detail_change",        # asks to change stored bank details
    "login_click_request",       # asks the reader to click through and sign in
    "secrecy_pressure",          # asks the reader not to tell anyone
    "authority_pressure",        # leans on rank or unavailability
    "consequence_threat",        # suspension, penalty, legal action
    "money_amount_present",      # a concrete sum is named
    "imperative_density",        # imperative verbs per sentence
    "link_count_norm",           # number of links, normalised
    "body_word_count_norm",      # very short mails are a phishing tell
]

# ----------------------------------------------------------------------
# Keyword and phrase families.
#
# These are matched on word boundaries, case insensitively, and each
# family saturates after FAMILY_SATURATION distinct hits: three separate
# urgency phrases is as urgent as a mail gets, and counting to nine would
# only reward verbosity.
# ----------------------------------------------------------------------

# Manufactured time pressure. The single most consistent feature of
# fraudulent mail, because an attacker needs the reader to act before
# they think or verify by phone.
URGENCY_TERMS: tuple[str, ...] = (
    "urgent", "urgently", "immediate", "immediately", "right away", "asap",
    "as soon as possible", "without delay", "time sensitive", "time-sensitive",
    "act now", "act fast", "act immediately", "quick action", "prompt action",
    "last reminder", "final reminder", "final notice", "last chance",
    "expires today", "expiring soon", "before close of business", "before eod",
    "by end of day", "top priority", "highest priority", "critical",
    "do not ignore", "cannot wait", "need this now", "treat as priority",
    "priority request", "emergency", "quickly please", "at the earliest",
)

# An explicit short deadline, which is stronger than a generic "urgent".
DEADLINE_PATTERNS: tuple[str, ...] = (
    r"\bwithin\s+\d{1,2}\s*(minutes?|mins?|hours?|hrs?)\b",
    r"\bin\s+the\s+next\s+\d{1,2}\s*(minutes?|hours?)\b",
    r"\bbefore\s+\d{1,2}\s*(am|pm|:\d{2})\b",
    r"\bby\s+(today|tonight|noon|eod|cob|close of business)\b",
    r"\bexpires?\s+(in|within)\s+\d{1,2}\b",
    r"\b(today|tonight)\s+(itself|without fail)\b",
    r"\bdeadline\s+is\s+(today|tomorrow)\b",
    r"\b24\s*(hours|hrs)\b",
    r"\b48\s*(hours|hrs)\b",
)

# Payment redirection: the money is real, the destination is not.
PAYMENT_REDIRECTION_PHRASES: tuple[str, ...] = (
    "new account details", "updated account details", "change of account",
    "change in bank", "new bank account", "new beneficiary", "beneficiary details",
    "remit to", "remit the payment", "wire the funds", "wire transfer",
    "transfer the balance", "process the payment to", "pay into", "pay to the account",
    "use the account below", "use the following account", "different account",
    "alternate account", "alternative account", "revised payment instructions",
    "updated payment instructions", "new payment instructions", "new iban",
    "new swift", "swift code", "ifsc code", "new ifsc", "account number below",
    "kindly process the transfer", "release the payment", "expedite the payment",
    "make the payment today", "hold the payment", "our old account is frozen",
    "our account is under audit", "bank has blocked our account",
)

# Invoice manipulation: the paperwork is rewritten so the fraud looks
# like routine accounts payable work.
INVOICE_MANIPULATION_PHRASES: tuple[str, ...] = (
    "revised invoice", "corrected invoice", "updated invoice", "reissued invoice",
    "amended invoice", "attached invoice", "outstanding invoice", "unpaid invoice",
    "overdue invoice", "duplicate invoice", "invoice correction", "credit note",
    "proforma invoice", "purchase order attached", "revised purchase order",
    "payment overdue", "pending payment", "balance due", "final invoice",
    "kindly settle", "settle the outstanding", "clear the outstanding",
    "invoice number has changed", "please disregard the previous invoice",
    "ignore the earlier invoice", "the earlier invoice was incorrect",
)

# Credential harvesting: the payload is the reader's password or OTP.
CREDENTIAL_HARVEST_PHRASES: tuple[str, ...] = (
    "verify your account", "verify your identity", "confirm your identity",
    "confirm your password", "update your password", "reset your password",
    "your password will expire", "password expires", "re-enter your password",
    "confirm your credentials", "validate your account", "reactivate your account",
    "unlock your account", "your account has been locked", "account suspended",
    "unusual sign-in", "unusual sign in", "unusual login", "suspicious login",
    "sign in to continue", "log in to continue", "login to view", "sign in to view",
    "click here to log in", "verify your mailbox", "mailbox is full",
    "mailbox quota", "revalidate your mailbox", "confirm your email address",
    "one time password", "otp", "enter the code", "share the code",
    "authentication code", "two factor code", "security code", "kyc verification",
    "complete your kyc", "update your kyc", "aadhaar verification",
    "net banking credentials", "card details", "cvv", "pin number",
)

# A bank detail change request, scored on its own because this one
# sentence is the payload of nearly every successful BEC. If it appears,
# an analyst should read the mail whatever else the model says.
BANK_DETAIL_CHANGE_PHRASES: tuple[str, ...] = (
    "change our bank details", "change the bank details", "update our bank details",
    "update the bank account", "update your records with our new",
    "change of bank account", "changed our bank", "changed our bankers",
    "we have switched banks", "new banking details", "amend our bank details",
    "update the beneficiary", "update our payment details",
    "note our new account", "kindly update your records",
    "future payments should be made to", "all future payments",
    "going forward please pay", "henceforth payments",
)

# "Click through and sign in", the mechanical core of credential theft.
LOGIN_CTA_PHRASES: tuple[str, ...] = (
    "click here", "click below", "click the link", "click this link",
    "follow the link", "use the link below", "tap here", "open the link",
    "sign in here", "log in here", "login here", "access your account here",
    "review the document here", "view the document", "view attachment online",
    "open the secure message", "secure message portal", "download the file here",
    "authenticate here", "continue to your account", "proceed to login",
    "verify here", "confirm here", "update here", "start verification",
)

# Secrecy pressure: isolates the reader from the colleague who would
# spot the fraud in four seconds.
SECRECY_PHRASES: tuple[str, ...] = (
    "keep this between us", "keep this confidential", "strictly confidential",
    "do not discuss", "do not share this", "do not tell", "no one else needs to know",
    "handle this discreetly", "discreet", "private and confidential",
    "do not copy anyone", "do not loop in", "without involving",
    "before informing the team", "this is sensitive", "internal and confidential",
)

# Authority pressure: rank plus unavailability, so the request cannot be
# checked by walking to someone's desk.
AUTHORITY_PHRASES: tuple[str, ...] = (
    "i am in a meeting", "i am travelling", "i am traveling", "i am on a flight",
    "cannot take calls", "unable to take calls", "do not call me",
    "reach me only by email", "on behalf of the ceo", "on behalf of the chairman",
    "on behalf of the director", "as the managing director", "as your ceo",
    "instruction from the chairman", "board instruction", "per the ceo",
    "authorised by the director", "i need you to handle", "i am counting on you",
    "trust you to handle this", "do this for me personally",
)

# Consequence threats: what allegedly happens if the reader thinks first.
CONSEQUENCE_PHRASES: tuple[str, ...] = (
    "account will be suspended", "account will be closed", "will be terminated",
    "will be deactivated", "service will be discontinued", "legal action",
    "penalty will apply", "late fee", "you will be charged", "loss of access",
    "permanent deletion", "data will be lost", "compliance violation",
    "regulatory action", "your salary will be withheld", "credit rating",
    "report to the authorities", "failure to comply",
)

# Imperative verbs, counted per sentence. Fraudulent mail is a list of
# instructions; ordinary business mail is mostly statements.
IMPERATIVE_VERBS: tuple[str, ...] = (
    "click", "verify", "confirm", "update", "send", "transfer", "remit",
    "pay", "process", "release", "download", "open", "review", "sign",
    "login", "log", "provide", "share", "reply", "respond", "call",
    "complete", "submit", "authorise", "authorize", "approve", "arrange",
    "ensure", "act", "proceed", "kindly", "please",
)

# Concrete sums. Naming an amount makes a request feel routine and
# actionable; it also tells the analyst what was at stake.
MONEY_PATTERNS: tuple[str, ...] = (
    r"(?:inr|rs\.?|usd|eur|gbp|aed|sgd)\s*[\d,]+(?:\.\d{1,2})?",
    r"[₹$€£]\s*[\d,]+(?:\.\d{1,2})?",
    r"\b[\d,]+(?:\.\d{1,2})?\s*(?:lakh|lakhs|crore|crores|million|mn|bn|k)\b",
    r"\b\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?\b",
)

# Distinct hits at which a family counts as fully present. Three separate
# phrases from one family is as strong as that family gets.
FAMILY_SATURATION: int = 3

_MODEL: Any = None
_TOKENIZER: Any = None
_MODEL_TRIED: bool = False
_PHRASE_REGEX_CACHE: dict[int, list[tuple[str, re.Pattern[str]]]] = {}


# ----------------------------------------------------------------------
# Matching helpers
# ----------------------------------------------------------------------
def _compiled_family(phrases: tuple[str, ...]) -> list[tuple[str, re.Pattern[str]]]:
    """Compile a phrase family once and cache it by identity.

    Word boundaries on both ends, so "otp" does not match "adopted" and
    "pin number" does not match "spinning".
    """
    key = id(phrases)
    cached = _PHRASE_REGEX_CACHE.get(key)
    if cached is not None:
        return cached
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for phrase in phrases:
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b"
        compiled.append((phrase, re.compile(pattern, re.IGNORECASE)))
    _PHRASE_REGEX_CACHE[key] = compiled
    return compiled


def family_score(text: str, phrases: tuple[str, ...]) -> tuple[float, list[str]]:
    """Saturating score for one phrase family, plus the phrases that hit."""
    if not text:
        return 0.0, []
    hits = [phrase for phrase, regex in _compiled_family(phrases) if regex.search(text)]
    return min(len(hits) / FAMILY_SATURATION, 1.0), hits


def _regex_family_score(text: str, patterns: tuple[str, ...]) -> tuple[float, list[str]]:
    """Saturating score for a family expressed directly as regexes."""
    if not text:
        return 0.0, []
    hits: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hits.append(match.group(0).strip())
    return min(len(hits) / FAMILY_SATURATION, 1.0), hits


def _imperative_density(text: str) -> float:
    """Imperative verbs per sentence, capped at 1.0.

    Counted per sentence rather than per word so that a long mail is not
    penalised for length and a two-line mail of pure instructions is not
    let off for brevity.
    """
    if not text:
        return 0.0
    sentences = [s for s in re.split(r"[.!?\n]+", text) if s.strip()]
    if not sentences:
        return 0.0
    verb_set = set(IMPERATIVE_VERBS)
    count = 0
    for sentence in sentences:
        words = re.findall(r"[a-z']+", sentence.lower())
        if any(word in verb_set for word in words[:4]):
            count += 1
    return min(count / len(sentences), 1.0)


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def _message_text(email: ParsedEmail) -> str:
    """Plain text body, falling back to tags stripped out of the HTML."""
    body = (email.text_body or "").strip()
    if body:
        return body
    html = email.html_body or ""
    if not html:
        return ""
    without_scripts = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", without_scripts)).strip()


def _extract_details(email: ParsedEmail) -> dict[str, Any]:
    """Features plus the matched phrases the evidence row prints."""
    subject = email.subject or ""
    body = _message_text(email)
    combined = f"{subject}\n{body}"

    urgency_body, urgency_body_hits = family_score(body, URGENCY_TERMS)
    urgency_subject, urgency_subject_hits = family_score(subject, URGENCY_TERMS)
    deadline, deadline_hits = _regex_family_score(combined, DEADLINE_PATTERNS)
    payment, payment_hits = family_score(combined, PAYMENT_REDIRECTION_PHRASES)
    invoice, invoice_hits = family_score(combined, INVOICE_MANIPULATION_PHRASES)
    credential, credential_hits = family_score(combined, CREDENTIAL_HARVEST_PHRASES)
    bank_change, bank_change_hits = family_score(combined, BANK_DETAIL_CHANGE_PHRASES)
    login_cta, login_cta_hits = family_score(combined, LOGIN_CTA_PHRASES)
    secrecy, secrecy_hits = family_score(combined, SECRECY_PHRASES)
    authority, authority_hits = family_score(combined, AUTHORITY_PHRASES)
    consequence, consequence_hits = family_score(combined, CONSEQUENCE_PHRASES)
    money, money_hits = _regex_family_score(combined, MONEY_PATTERNS)

    # A "click through and sign in" only counts as a login request if
    # there is somewhere to click. A call to action with no link is
    # usually a real document handover.
    link_count = len(email.urls or [])
    if link_count == 0:
        login_cta *= 0.4

    word_count = len(re.findall(r"\w+", body))
    # Very short mails score high here: a forty word mail with a link and
    # a deadline is the commonest phishing shape there is.
    short_body = 1.0 if word_count and word_count < 40 else max(0.0, min(1.0, 1.0 - word_count / 400.0))

    return {
        "urgency_body": urgency_body,
        "urgency_subject": urgency_subject,
        "deadline_pressure": deadline,
        "payment_redirection": payment,
        "invoice_manipulation": invoice,
        "credential_harvest": credential,
        "bank_detail_change": bank_change,
        "login_click_request": login_cta,
        "secrecy_pressure": secrecy,
        "authority_pressure": authority,
        "consequence_threat": consequence,
        "money_amount_present": 1.0 if money_hits else 0.0,
        "imperative_density": _imperative_density(body),
        "link_count_norm": min(link_count / 5.0, 1.0),
        "body_word_count_norm": short_body,
        "_text": combined,
        "_word_count": word_count,
        "_link_count": link_count,
        "_hits": {
            "urgency (subject)": urgency_subject_hits,
            "urgency (body)": urgency_body_hits,
            "deadline": deadline_hits,
            "payment redirection": payment_hits,
            "invoice manipulation": invoice_hits,
            "credential harvest": credential_hits,
            "bank detail change": bank_change_hits,
            "login call to action": login_cta_hits,
            "secrecy": secrecy_hits,
            "authority": authority_hits,
            "consequence threat": consequence_hits,
            "amount": money_hits,
        },
    }


def _extract_features(email: ParsedEmail) -> dict[str, float]:
    """Pull every feature in FEATURE_NAMES out of the email."""
    details = _extract_details(email)
    return {name: float(details[name]) for name in FEATURE_NAMES}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
# Heuristic weights. A bank detail change request and credential
# harvesting dominate because each is an end in itself; urgency and
# authority are multipliers of intent rather than intent on their own,
# so they are weighted lower and capped.
_H_WEIGHTS: dict[str, float] = {
    "bank_detail_change": 0.42,
    "credential_harvest": 0.36,
    "payment_redirection": 0.30,
    "login_click_request": 0.22,
    "invoice_manipulation": 0.18,
    "urgency_subject": 0.12,
    "urgency_body": 0.10,
    "deadline_pressure": 0.10,
    "secrecy_pressure": 0.14,
    "authority_pressure": 0.10,
    "consequence_threat": 0.10,
    "imperative_density": 0.08,
    "money_amount_present": 0.05,
    "link_count_norm": 0.04,
    "body_word_count_norm": 0.03,
}


def _heuristic_score(features: dict[str, float]) -> float:
    """Documented stand in until DistilBERT is fine tuned.

    A weighted sum over the phrase families, clamped to [0, 1], with one
    interaction term: a request to act plus manufactured urgency scores
    higher than the sum of the two, because the combination is the
    grammar of fraud rather than two independent oddities.

    Weights reflect what each family means on its own:

      * bank_detail_change (0.42) and credential_harvest (0.36) are
        complete attacks by themselves;
      * payment_redirection (0.30) and login_click_request (0.22) are the
        mechanics that carry them;
      * urgency, authority, secrecy and threats (0.10 to 0.14 each) are
        pressure, not payload. On their own they describe half of all
        legitimate escalation mail, which is why no combination of them
        alone can reach the BLOCK band here.

    Known weaknesses for whoever trains the real model to beat: this
    scorer has no notion of negation ("we will never ask you to confirm
    your password" reads as credential harvesting), no notion of
    quoting (a forwarded phishing mail being reported to IT scores like
    the original), and no coverage of any language other than English.
    A fine tuned DistilBERT fixes all three, which is the argument for
    training it.
    """
    score = sum(weight * features.get(name, 0.0) for name, weight in _H_WEIGHTS.items())
    action = max(
        features.get("bank_detail_change", 0.0),
        features.get("credential_harvest", 0.0),
        features.get("payment_redirection", 0.0),
        features.get("login_click_request", 0.0),
    )
    pressure = max(
        features.get("urgency_subject", 0.0),
        features.get("urgency_body", 0.0),
        features.get("deadline_pressure", 0.0),
        features.get("consequence_threat", 0.0),
    )
    score += 0.15 * action * pressure
    return max(0.0, min(1.0, score))


def _positive_label_index(model: Any) -> int:
    """Index of the fraud class in the classifier's label map.

    Reads config.id2label rather than assuming index 1, because a model
    trained by someone else may have ordered its labels either way and a
    silently inverted signal is worse than no signal.
    """
    try:
        id2label = getattr(model.config, "id2label", None) or {}
        for index, label in id2label.items():
            text = str(label).lower()
            if any(token in text for token in ("fraud", "phish", "malicious", "scam", "spam")):
                return int(index)
        if len(id2label) == 2 and "1" in {str(k) for k in id2label}:
            return 1
    except Exception:
        pass
    return 1


def _load_model() -> Any:
    """Load the trained model once, or return None if the slot is empty.

    Returns (tokenizer, model) so the caller does not have to know which
    of the two is missing. Loading happens on first use and never at
    import time: a transformer costs seconds and several hundred
    megabytes, and the CLI has to be able to print its help text.
    """
    global _MODEL, _TOKENIZER, _MODEL_TRIED
    if _MODEL_TRIED:
        return (_TOKENIZER, _MODEL) if _MODEL is not None else None
    _MODEL_TRIED = True
    if not MODEL_PATH or not os.path.isdir(MODEL_PATH):
        return None
    if not (HAS_TRANSFORMERS and HAS_TORCH):
        return None
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _MODEL.eval()
    except Exception:
        _TOKENIZER, _MODEL = None, None
        return None
    return (_TOKENIZER, _MODEL)


def _model_score(features: dict[str, float], text: str = "") -> float | None:
    """Score with the trained model if the slot is filled, else None.

    The extra `text` argument is a documented departure from the other
    signals: a sequence classifier consumes the message, not the tabular
    features. FEATURE_NAMES stays the contract for the heuristic, for the
    evidence row, and for comparing the two scorers on the same mail.
    """
    loaded = _load_model()
    if loaded is None:
        return None
    tokenizer, model = loaded
    if not text.strip():
        return None
    try:
        batch = tokenizer(
            text, truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**batch).logits[0]
        exponentials = [math.exp(float(value)) for value in logits]
        total = sum(exponentials) or 1.0
        probabilities = [value / total for value in exponentials]
        index = _positive_label_index(model)
        if index >= len(probabilities):
            index = len(probabilities) - 1
        return max(0.0, min(1.0, probabilities[index]))
    except Exception:
        return None


def _evidence_row(details: dict[str, Any]) -> str:
    """One short printable line naming the families that fired."""
    parts: list[str] = []
    for family, hits in details["_hits"].items():
        if not hits:
            continue
        sample = ", ".join(hits[:2])
        parts.append(f"{family}: {sample}")
    if not parts:
        return f"no fraudulent intent patterns matched in {details['_word_count']} words"
    row = "; ".join(parts[:5])
    if len(parts) > 5:
        row += f"; +{len(parts) - 5} more families"
    return row if len(row) <= 220 else row[:217] + "..."


def run(email: ParsedEmail) -> SignalResult:
    """Score what the message asks the reader to do. Never raises."""
    try:
        details = _extract_details(email)
        if not details["_text"].strip():
            return SignalResult(
                signal_id=SIGNAL_ID,
                name=SIGNAL_NAME,
                score=0.0,
                status="abstain",
                evidence_row="no subject or body text to read, intent not assessed",
                details={"reason": "empty message text"},
            )
        features = {name: float(details[name]) for name in FEATURE_NAMES}
        model_value = _model_score(features, details["_text"])
        score = model_value if model_value is not None else _heuristic_score(features)
        public: dict[str, Any] = {k: v for k, v in details.items() if not k.startswith("_")}
        public.update(
            matched_phrases={k: v for k, v in details["_hits"].items() if v},
            word_count=details["_word_count"],
            link_count=details["_link_count"],
            scorer="distilbert" if model_value is not None else "keyword-heuristic",
        )
        return SignalResult(
            signal_id=SIGNAL_ID,
            name=SIGNAL_NAME,
            score=round(float(max(0.0, min(1.0, score))), 4),
            status="ok",
            evidence_row=_evidence_row(details),
            details=public,
            model_backed=model_value is not None,
        )
    except Exception as exc:  # run() must never raise
        return SignalResult(
            signal_id=SIGNAL_ID,
            name=SIGNAL_NAME,
            score=0.0,
            status="abstain",
            evidence_row=f"intent signal could not run: {type(exc).__name__}",
            details={"error": repr(exc)},
        )
