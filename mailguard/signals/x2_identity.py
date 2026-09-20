"""x2 Identity: is the sender who the message claims the sender is?

This signal answers one question: does the identity presented in the
envelope and the visible headers hold up under inspection? It does not
read what the message asks for (that is x5) and it does not care whether
the message authenticated (that is x1). A business email compromise mail
routinely passes SPF, DKIM and DMARC, because the attacker really does
own `hdfc-verify.example` and really did sign for it. What the attacker
cannot do is make that domain be `hdfc.in`.

Five things are measured:

1. Cousin domain proximity to a list of protected domains, combining
   Unicode confusable analysis (characters that render identically but
   are different codepoints) with Levenshtein edit distance. Both are
   implemented here in pure Python, no external dependency.
2. Display name versus address disagreement. `"HDFC Bank"
   <accounts@hdfc-verify.example>` is the classic shape.
3. Reply-To pointing somewhere other than From, especially at free mail.
4. First contact: has this address ever written to this recipient before?
5. Thread hijack: a message claiming In-Reply-To a Message-ID that this
   installation has never seen. Faking a reply to a conversation that
   never happened is a strong indicator, because a real thread hijack
   requires mailbox access while a forged one costs nothing.

This signal carries most of the BEC detection in MailGuard, so it is
deliberately the most thorough of the four in this half of the project.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, Optional

# Allow `python mailguard/signals/x2_identity.py` to run the demo at the
# bottom of the file. When imported as part of the package (the normal
# case, via the signal loader) __package__ is set and this does nothing.
if __package__ in (None, ""):  # pragma: no cover - script convenience only
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from mailguard.core.models import ParsedEmail, SignalResult

SIGNAL_ID = "x2"
SIGNAL_NAME = "Identity"

# ======================================================================
# TRAINED MODEL SLOT
# ----------------------------------------------------------------------
# The model for this signal has NOT been trained yet.
# When it is, put the artefact path in MODEL_PATH below and the code
# will pick it up automatically on the next run. Until then the
# documented heuristic in _heuristic_score() runs instead, so the whole
# pipeline stays runnable end to end today.
#
# Expected artefact: a joblib dumped binary classifier (gradient
# boosting or plain logistic regression is plenty for fifteen tabular
# features) exposing predict_proba(X) -> array of shape (n, 2) where
# column 1 is P(the presented identity is forged). Train it on labelled
# BEC and brand impersonation corpora. The positive class is "this
# identity does not hold up", NOT "this mail is fraudulent": fusion owns
# the final call, and a signal that tries to make the verdict itself
# stops being separable and stops being explainable.
# Expected feature order is defined by FEATURE_NAMES below.
# ======================================================================
MODEL_PATH: str = ""          # <-- trained model path goes here

FEATURE_NAMES: list[str] = [
    "cousin_domain_score",          # 0..1 proximity to a protected domain
    "confusable_char_count",        # homoglyph codepoints in the domain
    "levenshtein_min_norm",         # 1 - (min edit distance / label length)
    "is_exact_protected_domain",    # 1.0 when the sender IS the real brand
    "tld_swap",                     # same brand label, different TLD
    "display_name_brand_mismatch",  # display claims a brand it cannot use
    "display_name_has_address",     # display name embeds a second address
    "reply_to_differs",             # Reply-To address != From address
    "reply_to_freemail",            # Reply-To lands at a free mail provider
    "reply_to_domain_differs",      # Reply-To domain != From domain
    "first_contact",                # no prior mail from this sender
    "sender_history_count_capped",  # prior messages seen, capped at 50
    "thread_hijack",                # In-Reply-To references an unknown ID
    "reply_without_references",     # claims a reply but carries no References
    "return_path_mismatch",         # Return-Path domain != From domain
]

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Protected domains are the brands this installation defends. An operator
# overrides them with MAILGUARD_PROTECTED_DOMAINS (comma separated) or
# MAILGUARD_PROTECTED_DOMAINS_FILE (one per line).
DEFAULT_PROTECTED_DOMAINS: list[str] = [
    "hdfc.in",
    "hdfcbank.com",
    "sbi.co.in",
    "onlinesbi.sbi",
    "icicibank.com",
    "axisbank.com",
    "kotak.com",
    "rbi.org.in",
    "npci.org.in",
    "gst.gov.in",
    "incometax.gov.in",
    "paypal.com",
    "stripe.com",
    "microsoft.com",
    "office365.com",
    "outlook.com",
    "google.com",
    "amazon.in",
    "apple.com",
    "adobe.com",
    "docusign.net",
    "dhl.com",
    "fedex.com",
    "linkedin.com",
]

# Display-name brand tokens mapped to the domains allowed to use them. A
# mail whose display name says "HDFC Bank" but whose address is not on
# the matching list is claiming an identity it does not hold.
BRAND_MAP: dict[str, tuple[str, ...]] = {
    "hdfc": ("hdfc.in", "hdfcbank.com"),
    "sbi": ("sbi.co.in", "onlinesbi.sbi", "statebank.in"),
    "state bank": ("sbi.co.in", "onlinesbi.sbi"),
    "icici": ("icicibank.com",),
    "axis bank": ("axisbank.com",),
    "kotak": ("kotak.com",),
    "reserve bank": ("rbi.org.in",),
    "income tax": ("incometax.gov.in",),
    "gst": ("gst.gov.in",),
    "paypal": ("paypal.com",),
    "stripe": ("stripe.com",),
    "microsoft": ("microsoft.com", "office365.com", "outlook.com"),
    "office 365": ("microsoft.com", "office365.com"),
    "google": ("google.com", "gmail.com"),
    "amazon": ("amazon.in", "amazon.com"),
    "apple": ("apple.com", "icloud.com"),
    "adobe": ("adobe.com",),
    "docusign": ("docusign.net", "docusign.com"),
    "dhl": ("dhl.com",),
    "fedex": ("fedex.com",),
    "linkedin": ("linkedin.com",),
    "netflix": ("netflix.com",),
}

# Free mail providers. A corporate brand asking for the reply at one of
# these is not a corporate brand.
FREEMAIL_DOMAINS: frozenset[str] = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in", "ymail.com",
    "hotmail.com", "outlook.com", "live.com", "msn.com", "aol.com",
    "protonmail.com", "proton.me", "gmx.com", "gmx.de", "mail.com",
    "yandex.com", "yandex.ru", "zoho.com", "icloud.com", "me.com",
    "rediffmail.com", "tutanota.com", "inbox.lv", "consultant.com",
    "post.com", "europe.com", "usa.com", "mail.ru",
})

# Two-part public suffixes, needed so that "sbi.co.in" splits to the
# label "sbi" rather than "co". This is not a full Public Suffix List,
# only the suffixes that matter for the protected domains above; swap in
# `publicsuffix2` if a deployment needs global coverage.
MULTI_PART_SUFFIXES: frozenset[str] = frozenset({
    "co.in", "net.in", "org.in", "gov.in", "ac.in", "res.in", "nic.in",
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "co.nz",
    "co.jp", "co.za", "com.br", "com.sg", "com.mx", "com.tr", "com.hk",
})

# Unicode confusables: a hand written subset of the Unicode confusables
# table covering the codepoints that actually turn up in registered
# cousin domains, mostly Cyrillic and Greek letters that render as Latin
# ones, plus a few fullwidth and script variants. Written as \u escapes
# so this source file stays pure ASCII on every platform.
CONFUSABLE_MAP: dict[str, str] = {
    "а": "a",  # CYRILLIC SMALL LETTER A
    "е": "e",  # CYRILLIC SMALL LETTER IE
    "о": "o",  # CYRILLIC SMALL LETTER O
    "р": "p",  # CYRILLIC SMALL LETTER ER
    "с": "c",  # CYRILLIC SMALL LETTER ES
    "у": "y",  # CYRILLIC SMALL LETTER U
    "х": "x",  # CYRILLIC SMALL LETTER HA
    "і": "i",  # CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    "ѕ": "s",  # CYRILLIC SMALL LETTER DZE
    "һ": "h",  # CYRILLIC SMALL LETTER SHHA
    "ј": "j",  # CYRILLIC SMALL LETTER JE
    "н": "h",  # CYRILLIC SMALL LETTER EN, renders as H
    "м": "m",  # CYRILLIC SMALL LETTER EM
    "т": "t",  # CYRILLIC SMALL LETTER TE
    "б": "b",  # CYRILLIC SMALL LETTER BE
    "α": "a",  # GREEK SMALL LETTER ALPHA
    "ο": "o",  # GREEK SMALL LETTER OMICRON
    "ρ": "p",  # GREEK SMALL LETTER RHO
    "ν": "v",  # GREEK SMALL LETTER NU
    "ι": "i",  # GREEK SMALL LETTER IOTA
    "κ": "k",  # GREEK SMALL LETTER KAPPA
    "ı": "i",  # LATIN SMALL LETTER DOTLESS I
    "‐": "-",  # HYPHEN
    "‑": "-",  # NON-BREAKING HYPHEN
    "ａ": "a",  # FULLWIDTH LATIN SMALL LETTER A
    "ｅ": "e",  # FULLWIDTH LATIN SMALL LETTER E
    "ｏ": "o",  # FULLWIDTH LATIN SMALL LETTER O
    "ɡ": "g",  # LATIN SMALL LETTER SCRIPT G
}

# Digit-for-letter substitution, the low tech cousin of a homoglyph
# ("hdfc-0nline" against "hdfc-online").
LEET_MAP: dict[str, str] = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
}

# Sender history. In production this is a database table (one row per
# sender per recipient, plus a Message-ID index); here it is a JSON file
# so the pipeline runs with no infrastructure at all. An empty path means
# "no history available", which is honest: every sender then reads as
# first contact and the evidence row says the store was empty, rather
# than implying the sender has a clean record.
#
# Expected shape:
#   {
#     "senders": {
#       "ceo@example.com": {
#         "count": 42,
#         "recipients": ["cfo@example.com"],
#         "first_seen": "2025-01-04T09:12:00",
#         "last_seen":  "2025-06-02T11:40:00",
#         "hours": [9, 10, 9, 11],
#         "bodies": ["... previous message text ..."],
#         "daily_counts": {"2025-06-02": 3}
#       }
#     },
#     "message_ids": ["<thread-root@example.com>"]
#   }
HISTORY_PATH: str = os.environ.get("MAILGUARD_HISTORY_PATH", "")

_MODEL: Any = None
_MODEL_TRIED: bool = False
_HISTORY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


# ----------------------------------------------------------------------
# Small pure-Python string tools
# ----------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings, iterative two-row DP.

    Implemented here on purpose: a C extension dependency for forty lines
    of dynamic programming is not worth the packaging cost, and a
    forensic tool should be auditable without reading a wheel.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (1 if ca != cb else 0)
            current[j] = min(insert, delete, substitute)
        previous = current
    return previous[len(b)]


def decode_idna(domain: str) -> str:
    """Return the Unicode form of a domain, decoding any xn-- labels.

    Punycode is how a homoglyph domain travels over SMTP, so comparing
    only the ASCII form would miss exactly the attack this module exists
    to find.
    """
    labels: list[str] = []
    for label in domain.split("."):
        if label.startswith("xn--"):
            try:
                labels.append(label.encode("ascii").decode("idna"))
                continue
            except Exception:
                pass
        labels.append(label)
    return ".".join(labels)


def skeleton(text: str) -> str:
    """Fold a string to its confusable skeleton: lowercase, ASCII-ish.

    NFKD removes the compatibility variants and the combining marks, then
    CONFUSABLE_MAP and LEET_MAP fold the remaining lookalikes. Two
    strings sharing a skeleton are visually interchangeable in a mail
    client, which is all an attacker needs.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    out: list[str] = []
    for ch in decomposed:
        if unicodedata.combining(ch):
            continue
        out.append(CONFUSABLE_MAP.get(ch, LEET_MAP.get(ch, ch)))
    return "".join(out)


def count_confusables(text: str) -> int:
    """Number of non-ASCII codepoints that fold to an ASCII lookalike."""
    total = 0
    for ch in text.lower():
        if ord(ch) <= 127:
            continue
        if ch in CONFUSABLE_MAP:
            total += 1
            continue
        folded = skeleton(ch)
        if folded and folded.isascii() and folded.isalnum():
            total += 1
    return total


def split_registrable(domain: str) -> tuple[list[str], str, str]:
    """Split a domain into (subdomain labels, registrable label, suffix)."""
    parts = [p for p in domain.lower().strip(".").split(".") if p]
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_PART_SUFFIXES:
        return parts[:-3], parts[-3], ".".join(parts[-2:])
    if len(parts) >= 2:
        return parts[:-2], parts[-2], parts[-1]
    return [], (parts[0] if parts else ""), ""


# ----------------------------------------------------------------------
# Cousin domain scoring
# ----------------------------------------------------------------------
def _compare_to_protected(
    candidate: str, protected: str
) -> tuple[float, str, Optional[int], bool]:
    """Score one candidate domain against one protected domain.

    Returns (score, mechanism, edit_distance, tld_swap). Each branch is a
    separate registration trick seen in the wild, and each gets its own
    score because they are not equally strong evidence.
    """
    cand_subs, cand_label, cand_suffix = split_registrable(candidate)
    prot_subs, prot_label, prot_suffix = split_registrable(protected)
    if not cand_label or not prot_label:
        return 0.0, "", None, False

    # The protected domain sits in the subdomain of something else:
    # hdfc.in.secure-login.example. The strongest trick, because the
    # brand renders verbatim in a mail client that truncates the host.
    joined_subs = ".".join(cand_subs)
    if prot_label in cand_subs or (protected and protected in joined_subs):
        return (
            0.97,
            "protected domain used as a subdomain of an unrelated registrable domain",
            None,
            False,
        )

    # Same registrable label, different suffix: hdfc.com against hdfc.in.
    if cand_label == prot_label and cand_suffix != prot_suffix:
        return 0.85, "same brand label registered under a different TLD", 0, True

    # Brand label embedded in a decorated label: hdfc-verify, secure-hdfc,
    # hdfc-in-support. A separator or extra token is required around the
    # brand, so that a four letter brand does not match every domain that
    # happens to contain those letters.
    if prot_label != cand_label and prot_label in cand_label and len(prot_label) >= 4:
        remainder = cand_label.replace(prot_label, "", 1)
        if re.fullmatch(r"[-_]?[a-z0-9-]{2,}", remainder or ""):
            return 0.90, f"brand label '{prot_label}' embedded in an unrelated domain", None, False
        return 0.70, f"brand label '{prot_label}' present with padding", None, False

    # Transposition of the brand label: hdcf.in against hdfc.in. Checked
    # separately from edit distance because an exact anagram of a short
    # brand is not a coincidence, while a two-edit neighbour of a short
    # brand very often is.
    if (
        cand_label != prot_label
        and len(prot_label) >= 4
        and sorted(cand_label) == sorted(prot_label)
    ):
        return 0.88, "brand label with two characters transposed", 2, cand_suffix != prot_suffix

    # Edit distance on the registrable label only: hdfcc.in, paypai.com.
    # Short labels are excluded because at length three or four almost
    # everything is within two edits of everything else.
    if len(prot_label) >= 5:
        dist = levenshtein(cand_label, prot_label)
        if dist == 1:
            return 0.93, "one character edit away from the brand label", 1, cand_suffix != prot_suffix
        if dist == 2 and len(prot_label) >= 7:
            return 0.78, "two character edits away from the brand label", 2, cand_suffix != prot_suffix
    return 0.0, "", None, False


def cousin_domain_score(
    domain: str, protected: Optional[list[str]] = None
) -> tuple[float, dict[str, Any]]:
    """Score how close `domain` sits to any protected domain.

    Returns (score in 0..1, details). A score of 0.0 together with
    details["exact_match"] true means the sender IS the protected brand,
    which is the opposite of suspicious and is reported as such.
    """
    details: dict[str, Any] = {
        "matched_domain": None,
        "mechanism": "",
        "edit_distance": None,
        "confusable_chars": 0,
        "tld_swap": False,
        "exact_match": False,
        "punycode": False,
    }
    domain = (domain or "").strip().strip(".").lower()
    if not domain:
        return 0.0, details

    details["punycode"] = "xn--" in domain
    unicode_form = decode_idna(domain)
    folded = skeleton(unicode_form)
    details["confusable_chars"] = count_confusables(unicode_form)

    candidates = [c for c in dict.fromkeys([domain, folded]) if c]
    protected_list = protected if protected is not None else load_protected_domains()

    best_score = 0.0
    for prot in protected_list:
        prot = prot.strip().lower()
        if not prot:
            continue
        # The domain itself, or a genuine subdomain of it (mail.hdfc.in).
        if domain == prot or domain.endswith("." + prot):
            details.update(
                exact_match=True,
                matched_domain=prot,
                mechanism=(
                    "exact match on a protected domain"
                    if domain == prot
                    else "subdomain of a protected domain"
                ),
                edit_distance=0,
            )
            return 0.0, details
        # Homoglyph: folds to the protected domain but is not it.
        if folded == prot and folded != domain:
            details.update(
                matched_domain=prot,
                mechanism="unicode confusable characters render as the protected domain",
                edit_distance=0,
            )
            return 0.99, details
        for cand in candidates:
            score, mechanism, dist, tld_swap = _compare_to_protected(cand, prot)
            if score > best_score:
                best_score = score
                details.update(
                    matched_domain=prot,
                    mechanism=mechanism,
                    edit_distance=dist,
                    tld_swap=tld_swap,
                )
    return best_score, details


# ----------------------------------------------------------------------
# Configuration and history loading
# ----------------------------------------------------------------------
def load_protected_domains() -> list[str]:
    """Protected domains from the environment, else the built-in list."""
    raw = os.environ.get("MAILGUARD_PROTECTED_DOMAINS", "")
    if raw.strip():
        return [d.strip().lower() for d in raw.split(",") if d.strip()]
    path = os.environ.get("MAILGUARD_PROTECTED_DOMAINS_FILE", "")
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return [
                    line.strip().lower()
                    for line in fh
                    if line.strip() and not line.lstrip().startswith("#")
                ]
        except OSError:
            pass
    return list(DEFAULT_PROTECTED_DOMAINS)


def _load_history(path: Optional[str] = None) -> dict[str, Any]:
    """Read the JSON sender history store, or return an empty one.

    Cached on (path, mtime) so a batch run does not re-read the file for
    every message. x4 keeps its own copy of this reader deliberately:
    each signal module has to stay independently loadable, and in
    production both would call one SenderHistoryRepository instead.
    """
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
# Display name analysis
# ----------------------------------------------------------------------
def _address_domain(address: Optional[str]) -> str:
    if not address or "@" not in address:
        return ""
    return address.rsplit("@", 1)[1].strip().strip(">").strip().lower()


def _bare_address(value: Optional[str]) -> str:
    """Pull `a@b.c` out of `"Name" <a@b.c>` without the email module."""
    if not value:
        return ""
    match = re.search(r"[\w.!#$%&'*+/=?^`{|}~-]+@[\w.-]+", value)
    return match.group(0).lower() if match else value.strip().lower()


def display_name_brand_mismatch(display: str, from_domain: str) -> tuple[float, str]:
    """Does the display name claim a brand the sending domain cannot use?"""
    text = (display or "").lower()
    if not text:
        return 0.0, ""
    from_domain = (from_domain or "").lower()
    for token, allowed in BRAND_MAP.items():
        if token in text:
            if any(from_domain == d or from_domain.endswith("." + d) for d in allowed):
                return 0.0, ""
            return 1.0, token
    return 0.0, ""


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def _extract_details(email: ParsedEmail) -> dict[str, Any]:
    """Every feature, plus the context the evidence row needs.

    Keys starting with an underscore are context for the analyst, not
    model features, and are stripped out of the feature vector.
    """
    from_domain = (email.from_domain or _address_domain(email.from_address)).lower()
    cousin, cousin_details = cousin_domain_score(from_domain)

    brand_mismatch, brand_token = display_name_brand_mismatch(email.from_display, from_domain)
    display_has_address = (
        1.0 if re.search(r"[\w.+-]+@[\w.-]+\.\w+", email.from_display or "") else 0.0
    )

    from_addr = _bare_address(email.from_address)
    reply_to_addr = _bare_address(email.reply_to)
    reply_to_domain = _address_domain(reply_to_addr)
    reply_differs = 1.0 if reply_to_addr and reply_to_addr != from_addr else 0.0
    reply_domain_differs = (
        1.0 if reply_to_domain and from_domain and reply_to_domain != from_domain else 0.0
    )
    reply_freemail = 1.0 if reply_differs and reply_to_domain in FREEMAIL_DOMAINS else 0.0

    history = _load_history()
    senders: dict[str, Any] = history.get("senders", {}) or {}
    record = senders.get(from_addr) or {}
    try:
        history_count = float(record.get("count", 0) or 0)
    except (TypeError, ValueError):
        history_count = 0.0
    # An empty history store is not the same thing as a known-bad sender,
    # but it is also not a clean record. We report first_contact and let
    # the evidence row say the store was empty.
    first_contact = 1.0 if history_count <= 0 else 0.0

    known_ids = {str(mid) for mid in (history.get("message_ids") or [])}
    in_reply_to = (email.in_reply_to or "").strip()
    thread_hijack = 1.0 if in_reply_to and known_ids and in_reply_to not in known_ids else 0.0
    reply_without_refs = 1.0 if in_reply_to and not email.references else 0.0

    return_path_domain = _address_domain(_bare_address(email.return_path))
    return_path_mismatch = (
        1.0 if return_path_domain and from_domain and return_path_domain != from_domain else 0.0
    )

    label_len = max(len(split_registrable(from_domain)[1]), 1)
    edit_distance = cousin_details.get("edit_distance")
    lev_norm = 0.0 if edit_distance is None else 1.0 - min(edit_distance / label_len, 1.0)

    return {
        "cousin_domain_score": cousin,
        "confusable_char_count": float(cousin_details["confusable_chars"]),
        "levenshtein_min_norm": lev_norm,
        "is_exact_protected_domain": 1.0 if cousin_details["exact_match"] else 0.0,
        "tld_swap": 1.0 if cousin_details["tld_swap"] else 0.0,
        "display_name_brand_mismatch": brand_mismatch,
        "display_name_has_address": display_has_address,
        "reply_to_differs": reply_differs,
        "reply_to_freemail": reply_freemail,
        "reply_to_domain_differs": reply_domain_differs,
        "first_contact": first_contact,
        "sender_history_count_capped": min(history_count, 50.0),
        "thread_hijack": thread_hijack,
        "reply_without_references": reply_without_refs,
        "return_path_mismatch": return_path_mismatch,
        "_from_domain": from_domain,
        "_matched_domain": cousin_details["matched_domain"],
        "_mechanism": cousin_details["mechanism"],
        "_punycode": cousin_details["punycode"],
        "_brand_token": brand_token,
        "_reply_to_domain": reply_to_domain,
        "_history_count": history_count,
        "_history_configured": bool(HISTORY_PATH),
        "_in_reply_to": in_reply_to,
    }


def _extract_features(email: ParsedEmail) -> dict[str, float]:
    """Pull every feature in FEATURE_NAMES out of the email."""
    details = _extract_details(email)
    return {name: float(details[name]) for name in FEATURE_NAMES}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
# Heuristic weights, deliberately additive and capped rather than
# multiplicative, so that one benign quirk on its own (first contact from
# a legitimate new supplier) cannot reach the BLOCK band, while any two
# structural findings together can.
_H_WEIGHTS: dict[str, float] = {
    "cousin_domain_score": 0.55,
    "display_name_brand_mismatch": 0.20,
    "thread_hijack": 0.35,
    "reply_to_freemail": 0.15,
    "reply_to_domain_differs": 0.07,
    "first_contact": 0.08,
    "display_name_has_address": 0.05,
    "return_path_mismatch": 0.05,
    "reply_without_references": 0.04,
}


def _heuristic_score(features: dict[str, float]) -> float:
    """Documented stand in until the model is trained.

    Rule of thumb: a weighted sum of the structural findings, clamped to
    [0, 1], with a homoglyph bonus and a credit for actually being the
    protected brand.

      * cousin domain proximity is the dominant term (0.55). A lookalike
        registration is a deliberate act with no innocent explanation.
      * a thread hijack claim is next (0.35): forging In-Reply-To is free
        for an attacker and never happens by accident.
      * a display name claiming a brand the domain cannot use adds 0.20.
      * Reply-To at free mail adds 0.15, bare Reply-To divergence only
        0.07, because mailing lists and ticketing systems do that
        legitimately all day.
      * first contact alone adds 0.08. On its own it must not reach WARN:
        every real new supplier is a first contact, and a system that
        flags all of them gets switched off.
      * being the exact protected domain subtracts 0.25, the only
        negative term here.

    Whoever trains the real classifier should beat this on precision at a
    fixed 0.1 percent false positive rate, measured on first-contact mail
    specifically, which is where this heuristic is weakest.
    """
    score = sum(weight * features.get(name, 0.0) for name, weight in _H_WEIGHTS.items())
    if features.get("confusable_char_count", 0.0) > 0:
        score += 0.10
    if features.get("is_exact_protected_domain", 0.0) > 0:
        score -= 0.25
    # A sender with a real correspondence record is unlikely to be an
    # impersonation. A compromised genuine account is x4's job, not this
    # signal's, which is exactly why x4 exists.
    if features.get("sender_history_count_capped", 0.0) >= 10:
        score -= 0.05
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
        import joblib  # optional dependency, imported lazily on purpose

        _MODEL = joblib.load(MODEL_PATH)
    except Exception:
        _MODEL = None
    return _MODEL


def _model_score(features: dict[str, float]) -> float | None:
    """Score with the trained model if the slot is filled, else None."""
    model = _load_model()
    if model is None:
        return None
    try:
        vector = [[features.get(name, 0.0) for name in FEATURE_NAMES]]
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(vector)[0][1])
        return float(model.predict(vector)[0])
    except Exception:
        return None


def _evidence_row(details: dict[str, Any]) -> str:
    """One short printable line an analyst reads."""
    parts: list[str] = []
    if details["cousin_domain_score"] > 0 and details["_matched_domain"]:
        parts.append(f"cousin domain of {details['_matched_domain']}, {details['_mechanism']}")
    elif details["is_exact_protected_domain"]:
        parts.append(f"genuine {details['_matched_domain']} domain")
    if details["_punycode"]:
        parts.append("punycode domain")
    if details["confusable_char_count"]:
        parts.append(f"{int(details['confusable_char_count'])} confusable characters in domain")
    if details["display_name_brand_mismatch"]:
        parts.append(f"display name claims '{details['_brand_token']}'")
    if details["thread_hijack"]:
        parts.append("replies to an unknown Message-ID")
    if details["reply_to_freemail"]:
        parts.append(f"reply-to free mail at {details['_reply_to_domain']}")
    elif details["reply_to_domain_differs"]:
        parts.append(f"reply-to {details['_reply_to_domain']}")
    if details["first_contact"]:
        parts.append(
            "first contact" if details["_history_configured"] else "first contact, no history store"
        )
    elif details["_history_count"]:
        parts.append(f"{int(details['_history_count'])} prior messages")
    if details["return_path_mismatch"]:
        parts.append("return-path domain differs")
    if not parts:
        parts.append("identity headers consistent")
    row = ", ".join(parts)
    return row if len(row) <= 220 else row[:217] + "..."


def run(email: ParsedEmail) -> SignalResult:
    """Score the sender identity. Never raises."""
    try:
        if not (email.from_address or "").strip():
            return SignalResult(
                signal_id=SIGNAL_ID,
                name=SIGNAL_NAME,
                score=0.0,
                status="abstain",
                evidence_row="no From address to evaluate, identity not assessed",
                details={"reason": "missing From address"},
            )
        details = _extract_details(email)
        features = {name: float(details[name]) for name in FEATURE_NAMES}
        model_value = _model_score(features)
        score = model_value if model_value is not None else _heuristic_score(features)
        public: dict[str, Any] = {k: v for k, v in details.items() if not k.startswith("_")}
        public.update(
            from_domain=details["_from_domain"],
            matched_domain=details["_matched_domain"],
            mechanism=details["_mechanism"],
            punycode=details["_punycode"],
            brand_token=details["_brand_token"],
            reply_to_domain=details["_reply_to_domain"],
            history_count=details["_history_count"],
            history_store_configured=details["_history_configured"],
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
            evidence_row=f"identity signal could not run: {type(exc).__name__}",
            details={"error": repr(exc)},
        )


if __name__ == "__main__":  # pragma: no cover - demonstration
    print("x2 Identity: cousin domain scorer, protected = ['hdfc.in', 'hdfcbank.com']")
    print("-" * 78)
    for candidate in [
        "hdfc-verify.example",
        "hdfc.in",
        "hdfcbank.com",
        "hdfc.com",
        "hdcf.in",
        "hdfc.in.secure-login.example",
        "example.org",
    ]:
        value, info = cousin_domain_score(candidate, protected=["hdfc.in", "hdfcbank.com"])
        note = info["mechanism"] or "no relation to a protected domain"
        print(f"{candidate:<32} score={value:.2f}  {note}")
    print("-" * 78)
    focus, focus_info = cousin_domain_score("hdfc-verify.example", protected=["hdfc.in"])
    print(f"hdfc-verify.example against hdfc.in -> {focus:.2f} ({focus_info['mechanism']})")
    print(f"levenshtein('hdfc-verify', 'hdfc') = {levenshtein('hdfc-verify', 'hdfc')}")
    # Same domain written with a Cyrillic 'c', which renders identically.
    homoglyph = "hdfс.in"
    h_score, h_info = cousin_domain_score(homoglyph, protected=["hdfc.in"])
    print(
        f"cyrillic lookalike (U+0441 in place of 'c') -> {h_score:.2f} "
        f"({h_info['confusable_chars']} confusable chars, {h_info['mechanism']})"
    )
