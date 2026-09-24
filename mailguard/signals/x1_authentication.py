"""x1 Authentication: SPF, DKIM, DMARC, and whether any of them vouch for the visible sender.

This signal contributes almost nothing when the attacker authenticates
correctly, and that is exactly why it cannot be the only signal. Someone
who registers `hdfc-verify.example` publishes perfect SPF and DKIM records
for it and passes every check: SPF and DKIM prove custody of a domain, not
the honesty of the person using it. x1 earns its place on the cases where
the checks FAIL, and on ALIGNMENT.

ALIGNMENT IS THE CHECK THAT MATTERS MOST. A message can pass SPF and DKIM
for `mailer-xyz.com` while the reader sees `From: cfo@yourbank.com`. Both
checks pass, and neither says anything about the address a human reads.
Alignment asks whether the domain that passed DKIM (the `d=` tag) and the
domain SPF checked (the envelope sender) are the visible From domain. When
neither is, the passes vouch for somebody else.

Sources, in order:

  1. The Authentication-Results header. One written by our own servers
     (authserv-id in TRUSTED_HOSTS) is preferred; if only a foreign one
     exists it is still read, but the evidence says it is unverified,
     because anyone can type an Authentication-Results header.
  2. If there is none, live verification: SPF with `pyspf` evaluated
     against the trust boundary IP (the one address we can stand behind),
     DKIM with `dkimpy`. Each sub check abstains when its library is
     missing or network lookups are disabled.

Also recorded: the DKIM key length (read from DNS with `dnspython`, when
available), and the DKIM selector in details["selector"], so the campaign
graph can link unrelated domains that reuse one selector, which is what a
single phishing kit deployed across many throwaway domains looks like.
DKIM key age is not published in DNS; it is recorded as None rather than
guessed.
"""
from __future__ import annotations

import base64
import re
from typing import Any, Optional

from mailguard.core.config import get_config, host_matches
from mailguard.core.models import ParsedEmail, SignalResult
from mailguard.forensics.trust_boundary import boundary_ip, boundary_host

SIGNAL_ID = "x1"
SIGNAL_NAME = "Authentication"

try:
    import spf  # pyspf

    HAS_SPF = True
except ImportError:
    HAS_SPF = False

try:
    import dkim  # dkimpy

    HAS_DKIM = True
except ImportError:
    HAS_DKIM = False

try:
    import dns.resolver

    HAS_DNS = True
except ImportError:
    HAS_DNS = False

try:
    import tldextract

    HAS_TLDEXTRACT = True
    _EXTRACT: Any = None
except ImportError:
    HAS_TLDEXTRACT = False
    _EXTRACT = None

# Risk per result, combined as a noisy OR. A pass contributes nothing.
SPF_RISK: dict[str, float] = {"pass": 0.0, "neutral": 0.10, "none": 0.15, "softfail": 0.25,
                              "fail": 0.40, "temperror": 0.10, "permerror": 0.20}
DKIM_RISK: dict[str, float] = {"pass": 0.0, "none": 0.15, "neutral": 0.10, "fail": 0.40,
                               "policy": 0.20, "temperror": 0.10, "permerror": 0.25}
DMARC_RISK: dict[str, float] = {"pass": 0.0, "bestguesspass": 0.0, "none": 0.15, "fail": 0.60,
                                "temperror": 0.10, "permerror": 0.20}
UNALIGNED_PASS_RISK: float = 0.45   # passes exist, but none of them for the From domain
WEAK_KEY_RISK: float = 0.20         # DKIM key under 1024 bits can be factored
MULTI_LABEL_SUFFIXES: frozenset[str] = frozenset(
    {"co.uk", "org.uk", "ac.uk", "com.au", "co.in", "net.in", "org.in", "ac.in", "gov.in",
     "co.jp", "com.br", "com.cn", "com.sg", "co.nz", "co.za"}
)

_COMMENT = re.compile(r"\([^()]*\)")
_METHOD = re.compile(r"^\s*([A-Za-z][\w-]*)\s*=\s*([A-Za-z]+)")
_PROP = re.compile(r"([A-Za-z][\w-]*)\.([A-Za-z][\w-]*)\s*=\s*(\"[^\"]*\"|[^\s;]+)")


def org_domain(domain: Optional[str]) -> str:
    """Registrable domain: `mail.hdfc-verify.example` -> `hdfc-verify.example`."""
    value = (domain or "").strip().lower().rstrip(".")
    if "@" in value:
        value = value.rsplit("@", 1)[1]
    if not value:
        return ""
    if HAS_TLDEXTRACT:
        global _EXTRACT
        try:
            if _EXTRACT is None:
                _EXTRACT = tldextract.TLDExtract(suffix_list_urls=())  # offline
            parts = _EXTRACT(value)
            if parts.domain and parts.suffix:
                return f"{parts.domain}.{parts.suffix}"
        except Exception:
            pass
    labels = value.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in MULTI_LABEL_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def parse_auth_results(value: str) -> dict[str, Any]:
    """Parse one Authentication-Results header (RFC 8601) into authserv-id and results."""
    text = value or ""
    for _ in range(3):
        text = _COMMENT.sub(" ", text)
    parts = [p.strip() for p in text.split(";")]
    authserv = parts[0].split()[0].lower() if parts and parts[0].split() else ""
    results: list[dict[str, Any]] = []
    for part in parts[1:]:
        match = _METHOD.match(part)
        if match:
            props = {f"{m.group(1).lower()}.{m.group(2).lower()}": m.group(3).strip('"') for m in _PROP.finditer(part)}
            results.append({"method": match.group(1).lower(), "result": match.group(2).lower(), "props": props})
    return {"authserv_id": authserv, "results": results}


def dkim_tags(signature: str) -> dict[str, str]:
    """Tags of one DKIM-Signature header (whitespace removed from values)."""
    tags: dict[str, str] = {}
    for token in str(signature or "").split(";"):
        if "=" in token:
            key, _, value = token.partition("=")
            tags[key.strip().lower()] = re.sub(r"\s+", "", value)
    return tags


def _rsa_bits_from_spki(der: bytes) -> Optional[int]:
    """Modulus size of an RSA SubjectPublicKeyInfo (or bare RSAPublicKey), by minimal DER walking."""

    def read(buf: bytes, pos: int) -> tuple[int, int, int]:
        tag = buf[pos]
        length = buf[pos + 1]
        pos += 2
        if length & 0x80:
            count = length & 0x7F
            length = int.from_bytes(buf[pos:pos + count], "big")
            pos += count
        return tag, pos, length

    try:
        tag, pos, _ = read(der, 0)                     # outer SEQUENCE
        if tag != 0x30:
            return None
        tag, inner, length = read(der, pos)
        if tag == 0x30:                                # AlgorithmIdentifier: SPKI form
            pos = inner + length
            tag, pos, length = read(der, pos)          # BIT STRING
            if tag != 0x03:
                return None
            pos += 1                                   # unused-bits byte
            tag, pos, _ = read(der, pos)               # RSAPublicKey SEQUENCE
            tag, pos, length = read(der, pos)          # INTEGER n
        elif tag == 0x02:                              # bare RSAPublicKey
            pos, length = inner, length
        else:
            return None
        modulus = der[pos:pos + length].lstrip(b"\x00")
        return len(modulus) * 8
    except (IndexError, ValueError):
        return None


def dkim_key_bits(domain: str, selector: str) -> Optional[int]:
    """DKIM public key size from DNS, or None when it cannot be resolved."""
    config = get_config()
    if not (HAS_DNS and config.network_lookups and domain and selector):
        return None
    try:
        resolver = dns.resolver.Resolver()
        resolver.lifetime = config.network_timeout_s
        answer = resolver.resolve(f"{selector}._domainkey.{domain}", "TXT")
        record = "".join(b"".join(r.strings).decode("ascii", "replace") for r in answer)
        tags = dkim_tags(record)
        if tags.get("k", "rsa").lower() == "ed25519":
            return 256
        if not tags.get("p"):
            return None  # empty p= means the key was revoked
        return _rsa_bits_from_spki(base64.b64decode(tags["p"] + "=="))
    except Exception:
        return None


def _from_headers(email: ParsedEmail, trusted_hosts: list[str]) -> Optional[dict[str, Any]]:
    """Results from Authentication-Results, preferring one written by our own servers."""
    values = email.headers.get("authentication-results", []) or []
    if not values:
        return None
    parsed = [parse_auth_results(v) for v in values]
    ours = [p for p in parsed if host_matches(p["authserv_id"], trusted_hosts)]
    chosen = ours or parsed
    found: dict[str, Any] = {
        "source": "Authentication-Results",
        "authserv_id": chosen[0]["authserv_id"],
        "trusted_source": bool(ours),
    }
    for header in chosen:
        for result in header["results"]:
            method, value, props = result["method"], result["result"], result["props"]
            if method == "spf" and "spf" not in found:
                found["spf"] = value
                mailfrom = props.get("smtp.mailfrom") or props.get("smtp.helo") or ""
                found["spf_domain"] = mailfrom.rsplit("@", 1)[-1].lower() if mailfrom else ""
            elif method == "dkim" and (found.get("dkim") != "pass"):
                found["dkim"] = value
                found["dkim_domain"] = (props.get("header.d") or props.get("header.i", "").rsplit("@", 1)[-1]).lower()
                if props.get("header.s"):
                    found["selector"] = props["header.s"]
            elif method == "dmarc" and "dmarc" not in found:
                found["dmarc"] = value
    return found


def _live(email: ParsedEmail) -> dict[str, Any]:
    """SPF against the boundary IP and DKIM with dkimpy, where libraries and policy allow."""
    config = get_config()
    found: dict[str, Any] = {"source": "live verification", "trusted_source": True, "skipped": []}
    if not config.network_lookups:
        found["skipped"].append("network lookups disabled")
        return found
    ip = boundary_ip(email.received_chain, email.trust_boundary_index)
    mailfrom = email.return_path or email.from_address
    if not HAS_SPF:
        found["skipped"].append("spf: pyspf not installed")
    elif not ip:
        found["skipped"].append("spf: no boundary IP to evaluate")
    else:
        try:
            result, _ = spf.check2(i=ip, s=mailfrom, h=boundary_host(email.received_chain, email.trust_boundary_index) or "unknown",
                                   timeout=config.network_timeout_s)
            found["spf"] = str(result).lower()
            found["spf_domain"] = mailfrom.rsplit("@", 1)[-1].lower() if "@" in mailfrom else ""
        except Exception as exc:
            found["skipped"].append(f"spf: {type(exc).__name__}")
    if not email.dkim_signatures:
        found["dkim"] = "none"
    elif not HAS_DKIM:
        found["skipped"].append("dkim: dkimpy not installed")
    else:
        try:
            found["dkim"] = "pass" if dkim.verify(email.raw_bytes) else "fail"
        except Exception as exc:
            found["dkim"] = "permerror"
            found["skipped"].append(f"dkim: {type(exc).__name__}")
    return found


def run(email: ParsedEmail) -> SignalResult:
    """Score authentication and alignment. Never raises."""
    try:
        config = get_config()
        found = _from_headers(email, config.trusted_hosts) or _live(email)

        # Selector and signing domain from the DKIM-Signature header itself.
        signature_tags = [dkim_tags(s) for s in email.dkim_signatures or []]
        from_org = org_domain(email.from_domain)
        chosen_sig = next((t for t in signature_tags if org_domain(t.get("d")) == from_org), None) or (
            signature_tags[0] if signature_tags else None
        )
        if chosen_sig:
            found.setdefault("selector", chosen_sig.get("s"))
            if not found.get("dkim_domain"):
                found["dkim_domain"] = (chosen_sig.get("d") or "").lower()

        if not any(found.get(k) for k in ("spf", "dkim", "dmarc")):
            reason = "no Authentication-Results header and live verification unavailable"
            if found.get("skipped"):
                reason += " (" + "; ".join(found["skipped"]) + ")"
            return SignalResult(SIGNAL_ID, SIGNAL_NAME, 0.0, "abstain", reason,
                                details={"selector": found.get("selector"), "abstain_reason": reason})

        spf_aligned = bool(found.get("spf_domain")) and org_domain(found.get("spf_domain")) == from_org
        dkim_aligned = bool(found.get("dkim_domain")) and org_domain(found.get("dkim_domain")) == from_org
        spf_pass = found.get("spf") == "pass"
        dkim_pass = found.get("dkim") == "pass"

        risks: list[tuple[float, str]] = []
        for key, table, label in (("spf", SPF_RISK, "SPF"), ("dkim", DKIM_RISK, "DKIM"), ("dmarc", DMARC_RISK, "DMARC")):
            value = found.get(key)
            if value and table.get(value, 0.15) > 0:
                risks.append((table.get(value, 0.15), f"{label} {value}"))
        if (spf_pass or dkim_pass) and not ((spf_pass and spf_aligned) or (dkim_pass and dkim_aligned)):
            risks.append((UNALIGNED_PASS_RISK, "passes are for a domain other than the From domain"))

        key_bits = dkim_key_bits(found.get("dkim_domain") or "", found.get("selector") or "")
        if key_bits is not None and key_bits < 1024:
            risks.append((WEAK_KEY_RISK, f"DKIM key only {key_bits} bits"))

        remaining = 1.0
        for risk, _label in risks:
            remaining *= 1.0 - risk
        score = 1.0 - remaining

        pieces: list[str] = []
        if found.get("spf"):
            pieces.append(f"SPF {found['spf']}" + ("" if not spf_pass else (" aligned" if spf_aligned else f" for {found.get('spf_domain')}")))
        if found.get("dkim"):
            detail = f" (d={found['dkim_domain']}" + (f", s={found['selector']}" if found.get("selector") else "") + ")" if found.get("dkim_domain") else ""
            pieces.append(f"DKIM {found['dkim']}{detail}" + (", d= mismatch with From domain" if dkim_pass and not dkim_aligned else ""))
        if found.get("dmarc"):
            pieces.append(f"DMARC {found['dmarc']}")
        if not found.get("trusted_source"):
            pieces.append("from an Authentication-Results header our servers did not write")
        row = ", ".join(pieces)

        return SignalResult(
            signal_id=SIGNAL_ID,
            name=SIGNAL_NAME,
            score=round(max(0.0, min(1.0, score)), 4),
            status="ok",
            evidence_row=row if len(row) <= 200 else row[:197] + "...",
            details={
                "spf": found.get("spf"),
                "dkim": found.get("dkim"),
                "dmarc": found.get("dmarc"),
                "spf_domain": found.get("spf_domain"),
                "dkim_domain": found.get("dkim_domain"),
                "from_domain": email.from_domain,
                "spf_aligned": spf_aligned,
                "dkim_aligned": dkim_aligned,
                "selector": found.get("selector"),
                "dkim_selector": found.get("selector"),
                "dkim_key_bits": key_bits,
                "dkim_key_age_days": None,  # not published in DNS
                "source": found.get("source"),
                "authserv_id": found.get("authserv_id"),
                "trusted_source": found.get("trusted_source"),
                "risk_reasons": [label for _risk, label in risks],
                "skipped": found.get("skipped", []),
            },
        )
    except Exception as exc:  # run() must never raise
        return SignalResult(SIGNAL_ID, SIGNAL_NAME, 0.0, "abstain",
                            f"authentication signal could not run: {type(exc).__name__}", details={"error": repr(exc)})
