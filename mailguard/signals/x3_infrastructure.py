"""x3 Infrastructure: how old is the sending domain, and has it ever been set up to send mail?

Fraud runs on disposable infrastructure. A lookalike domain is typically
registered days before the attack, pointed at a bulk sender, and abandoned
a week later. Legitimate correspondents write from domains that are years
old and that have received mail for years. This signal measures that gap:

  domain age        days since registration, via RDAP (https://rdap.org/domain/<d>)
                    with `requests`, falling back to `python-whois`
  registrar         the registrar name, when RDAP or WHOIS gives one
  MX records        via `dnspython`. A domain with no MX has never been set
                    up to receive mail, so it has never been a real
                    correspondent's domain.
  name servers      the NS set; a domain with none is not live
  IP reputation     AbuseIPDB confidence for the trust boundary IP, only when
                    ABUSEIPDB_KEY is configured

Every lookup is optional, time boxed and wrapped: a missing library, a
disabled network, a reserved test domain or a failed request makes that
sub check "unknown", never an error. When nothing at all could be looked
up the signal ABSTAINS, because "could not measure" is not "measured clean".
"""
from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime, timezone
from typing import Any, Optional

from mailguard.core.config import get_config
from mailguard.core.models import ParsedEmail, SignalResult
from mailguard.forensics.trust_boundary import boundary_ip, is_internal_ip

SIGNAL_ID = "x3"
SIGNAL_NAME = "Infrastructure"

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import whois  # python-whois

    HAS_WHOIS = True
except ImportError:
    HAS_WHOIS = False

try:
    import dns.resolver

    HAS_DNS = True
except ImportError:
    HAS_DNS = False

# ======================================================================
# TRAINED MODEL SLOT
# ----------------------------------------------------------------------
# The model for this signal has NOT been trained yet.
# When it is, put the artefact path in MODEL_PATH below and the code
# will pick it up automatically on the next run. Until then the
# documented heuristic in _heuristic_score() runs instead, so the whole
# pipeline stays runnable end to end today.
#
# Expected artefact: joblib dump of a fitted estimator exposing
#   predict_proba(X) -> array of shape (n_samples, 2)
# Expected feature order is defined by FEATURE_NAMES below.
# ======================================================================
MODEL_PATH: str = ""          # <-- trained model path goes here

FEATURE_NAMES: list[str] = [
    "domain_age_days",       # days from registration to receipt, capped at 3650; -1 when unknown
    "domain_age_known",      # 1.0 when a registration date was found
    "domain_under_30d",      # 1.0 when the domain was under 30 days old at receipt
    "has_mx",                # 1.0 when the domain publishes MX records
    "mx_known",              # 1.0 when the MX lookup completed
    "ns_count",              # number of name servers, capped at 10
    "registrar_known",       # 1.0 when a registrar name was found
    "abuse_confidence",      # AbuseIPDB confidence for the boundary IP, 0..1
    "abuse_known",           # 1.0 when AbuseIPDB answered
]

# Reserved names (RFC 2606 / 6761) can never be registered or resolved, so
# they are never looked up. Fixtures use them.
RESERVED_TLDS: tuple[str, ...] = (".example", ".test", ".invalid", ".localhost", ".local")

RDAP_URL = "https://rdap.org/domain/{domain}"
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"

_MODEL = None
_CACHE: dict[str, dict[str, Any]] = {}


def _load_model():
    """Load the trained model once, or return None if the slot is empty."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not MODEL_PATH:
        return None
    try:
        import joblib
        _MODEL = joblib.load(MODEL_PATH)
        return _MODEL
    except Exception:
        return None


def _timeout() -> float:
    """Per request timeout: inside the signal's own time budget."""
    config = get_config()
    return max(0.1, min(config.network_timeout_s, config.signal_timeout_ms / 1000.0 * 0.45))


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, list):
        dates = [d for d in (_parse_date(v) for v in value) if d]
        return min(dates) if dates else None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _get_json(url: str, headers: dict[str, str], params: Optional[dict[str, str]] = None) -> Optional[dict[str, Any]]:
    """GET a JSON document with `requests`, or urllib when requests is missing."""
    try:
        if HAS_REQUESTS:
            response = requests.get(url, headers=headers, params=params, timeout=_timeout())
            if response.status_code != 200:
                return None
            return response.json()
        if params:
            from urllib.parse import urlencode

            url = f"{url}?{urlencode(params)}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=_timeout()) as response:
            return json.loads(response.read(2_000_000).decode("utf-8", "replace"))
    except Exception:
        return None


def domain_registration(domain: str) -> dict[str, Any]:
    """{created, registrar, source} from RDAP, then python-whois. Empty values when unknown."""
    info: dict[str, Any] = {"created": None, "registrar": None, "source": None}
    data = _get_json(RDAP_URL.format(domain=domain), {"Accept": "application/rdap+json", "User-Agent": "MailGuard-AI"})
    if data:
        for event in data.get("events", []) or []:
            if str(event.get("eventAction", "")).lower() == "registration":
                info["created"] = _parse_date(event.get("eventDate"))
        for entity in data.get("entities", []) or []:
            if "registrar" in (entity.get("roles") or []):
                vcard = entity.get("vcardArray") or []
                for item in (vcard[1] if len(vcard) > 1 else []):
                    if isinstance(item, list) and item and item[0] == "fn":
                        info["registrar"] = str(item[-1])
        if info["created"] or info["registrar"]:
            info["source"] = "rdap"
            return info
    if HAS_WHOIS:
        try:
            record = whois.whois(domain)
            info["created"] = _parse_date(getattr(record, "creation_date", None))
            registrar = getattr(record, "registrar", None)
            info["registrar"] = str(registrar) if registrar else None
            if info["created"] or info["registrar"]:
                info["source"] = "whois"
        except Exception:
            pass
    return info


def dns_records(domain: str) -> dict[str, Any]:
    """{has_mx, mx_known, ns: [...]} from dnspython. Unknown when unavailable."""
    info: dict[str, Any] = {"has_mx": None, "mx_known": False, "ns": [], "ns_known": False}
    if not HAS_DNS:
        return info
    resolver = dns.resolver.Resolver()
    resolver.lifetime = _timeout()
    try:
        answer = resolver.resolve(domain, "MX")
        info["has_mx"] = len(answer) > 0
        info["mx_known"] = True
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
        info["has_mx"] = False
        info["mx_known"] = True
    except Exception:
        pass
    try:
        answer = resolver.resolve(domain, "NS")
        info["ns"] = sorted(str(r.target).rstrip(".").lower() for r in answer)
        info["ns_known"] = True
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
        info["ns_known"] = True
    except Exception:
        pass
    return info


def abuse_confidence(ip: Optional[str]) -> Optional[float]:
    """AbuseIPDB confidence (0..1) for an IP, or None when not configured or unavailable."""
    key = get_config().abuseipdb_key
    if not key or not ip or is_internal_ip(ip):
        return None
    data = _get_json(ABUSEIPDB_URL, {"Key": key, "Accept": "application/json"}, {"ipAddress": ip, "maxAgeInDays": "90"})
    try:
        return float(data["data"]["abuseConfidenceScore"]) / 100.0 if data else None
    except (KeyError, TypeError, ValueError):
        return None


def _receipt_date(email: ParsedEmail) -> date:
    """When our infrastructure received the message: the top hop's timestamp, else Date, else today."""
    for hop in email.received_chain or []:
        if hop.trusted and hop.timestamp:
            return hop.timestamp.date()
    if email.date:
        return email.date.date()
    return datetime.now(timezone.utc).date()


def _lookup(email: ParsedEmail) -> dict[str, Any]:
    """Every raw fact this signal uses, cached per domain for the process."""
    config = get_config()
    domain = (email.from_domain or "").lower()
    facts: dict[str, Any] = {"domain": domain, "skipped": []}
    if not domain:
        facts["skipped"].append("no From domain")
    elif domain.endswith(RESERVED_TLDS):
        facts["skipped"].append(f"{domain} is a reserved test domain and cannot be looked up")
    elif not config.network_lookups:
        facts["skipped"].append("network lookups disabled")
    else:
        if domain not in _CACHE:
            _CACHE[domain] = {"registration": domain_registration(domain), "dns": dns_records(domain)}
        facts.update(_CACHE[domain])
        if not HAS_DNS:
            facts["skipped"].append("dnspython not installed")
    ip = boundary_ip(email.received_chain, email.trust_boundary_index)
    facts["boundary_ip"] = ip
    facts["abuse_confidence"] = abuse_confidence(ip) if config.network_lookups else None
    if not config.abuseipdb_key:
        facts["skipped"].append("AbuseIPDB key not set")
    return facts


def _extract_features(email: ParsedEmail) -> dict[str, float]:
    """Pull every feature in FEATURE_NAMES out of the email."""
    facts = email.meta.get("x3_facts") or _lookup(email)
    email.meta["x3_facts"] = facts
    registration = facts.get("registration") or {}
    records = facts.get("dns") or {}
    created = registration.get("created")
    age = (_receipt_date(email) - created).days if isinstance(created, date) else None
    email.meta["x3_domain_age_days"] = age  # uncapped, for the evidence row
    abuse = facts.get("abuse_confidence")
    return {
        "domain_age_days": float(min(age, 3650)) if age is not None else -1.0,
        "domain_age_known": 1.0 if age is not None else 0.0,
        "domain_under_30d": 1.0 if age is not None and age < 30 else 0.0,
        "has_mx": 1.0 if records.get("has_mx") else 0.0,
        "mx_known": 1.0 if records.get("mx_known") else 0.0,
        "ns_count": float(min(len(records.get("ns") or []), 10)),
        "registrar_known": 1.0 if registration.get("registrar") else 0.0,
        "abuse_confidence": float(abuse) if abuse is not None else 0.0,
        "abuse_known": 1.0 if abuse is not None else 0.0,
    }


def _heuristic_score(features: dict[str, float]) -> float:
    """Documented stand in until the model is trained.

    Rule of thumb, combined as a noisy OR so independent findings add up
    without any one of them saturating the score:

      under 30 days old AND no MX records    0.90  the disposable domain profile:
                                                   registered for this attack and
                                                   never set up as a real mailbox
      under 30 days old                      0.60
      30 to 90 days old                      0.30
      no MX records (MX lookup completed)    0.35  never a genuine correspondent
      no name servers (NS lookup completed)  0.20  not a live domain
      AbuseIPDB confidence c                 0.60 * c for the boundary IP

    What the trained model has to beat: this treats age as a step function,
    ignores the registrar entirely, and cannot tell a brand new but honest
    startup from a throwaway domain.
    """
    risks: list[float] = []
    known_age = features["domain_age_known"] > 0
    age = features["domain_age_days"]
    no_mx = features["mx_known"] > 0 and features["has_mx"] == 0
    if known_age and age < 30 and no_mx:
        risks.append(0.90)
    elif known_age and age < 30:
        risks.append(0.60)
    elif known_age and age < 90:
        risks.append(0.30)
    if no_mx and not (known_age and age < 30):
        risks.append(0.35)
    if features["mx_known"] > 0 and features["ns_count"] == 0:
        risks.append(0.20)
    if features["abuse_known"] > 0:
        risks.append(0.60 * features["abuse_confidence"])
    remaining = 1.0
    for risk in risks:
        remaining *= 1.0 - max(0.0, min(1.0, risk))
    return 1.0 - remaining


def _model_score(features: dict[str, float]) -> float | None:
    """Score with the trained model if the slot is filled, else None."""
    model = _load_model()
    if model is None:
        return None
    try:
        row = [[float(features[name]) for name in FEATURE_NAMES]]
        return float(model.predict_proba(row)[0][1])
    except Exception:
        return None


def run(email: ParsedEmail) -> SignalResult:
    """Score the sending infrastructure. Never raises."""
    try:
        features = _extract_features(email)
        facts = email.meta.get("x3_facts") or {}
        if not (features["domain_age_known"] or features["mx_known"] or features["abuse_known"]):
            reason = "infrastructure not assessable: " + ("; ".join(facts.get("skipped") or []) or "every lookup failed")
            return SignalResult(SIGNAL_ID, SIGNAL_NAME, 0.0, "abstain", reason[:200],
                                details={"features": features, "abstain_reason": reason})
        model_value = _model_score(features)
        score = model_value if model_value is not None else _heuristic_score(features)

        registration = facts.get("registration") or {}
        records = facts.get("dns") or {}
        pieces: list[str] = [facts.get("domain", "")]
        real_age = email.meta.get("x3_domain_age_days")
        if real_age is not None:
            pieces.append(f"registered {real_age} days before receipt")
        if registration.get("registrar"):
            pieces.append(f"via {registration['registrar']}")
        if features["mx_known"]:
            pieces.append("MX present" if features["has_mx"] else "no MX records")
        if features["abuse_known"]:
            pieces.append(f"AbuseIPDB {int(features['abuse_confidence'] * 100)}% for {facts.get('boundary_ip')}")
        row = ", ".join(p for p in pieces if p)
        return SignalResult(
            signal_id=SIGNAL_ID,
            name=SIGNAL_NAME,
            score=round(max(0.0, min(1.0, score)), 4),
            status="ok",
            evidence_row=row[:200],
            details={
                "features": features,
                "registrar": registration.get("registrar"),
                "registration_source": registration.get("source"),
                "name_servers": records.get("ns") or [],
                "domain_age_days": real_age,
                "skipped": facts.get("skipped") or [],
                "scorer": "model" if model_value is not None else "heuristic",
            },
            model_backed=model_value is not None,
        )
    except Exception as exc:  # run() must never raise
        return SignalResult(SIGNAL_ID, SIGNAL_NAME, 0.0, "abstain",
                            f"infrastructure signal could not run: {type(exc).__name__}", details={"error": repr(exc)})
