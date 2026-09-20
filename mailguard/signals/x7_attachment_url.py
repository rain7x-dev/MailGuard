"""x7 Attachment and URL: what is the payload, and where does it point?

x5 reads the request; this signal inspects the things the request wants
the reader to open. Two halves, scored together:

URL structure. Punycode in the host, excessive subdomain depth, digits
standing in for letters, link shorteners, anchor text that disagrees with
its href, link domains unrelated to the sending domain, raw IP hosts,
credential-flavoured paths, and the small print of a URL that a reader
never sees (userinfo before an @, a non standard port, absurd length).

Attachment risk. Macro bearing Office documents, detected two ways: the
OLE2 magic \\xd0\\xcf\\x11\\xe0 for the legacy binary formats, and the
presence of vbaProject.bin inside the zip for OOXML. Plus MIME type
against actual magic bytes, double extensions, right-to-left override
tricks, executables and archives.

Redirect resolution is available and OFF by default. See
RESOLVE_REDIRECTS below for why: fetching an attacker supplied URL from
the analysis host is an outbound action taken on the attacker's
instruction, and this tool does not do that unless an operator says so.

Two trained model slots live in this file, one per half: a character
level CNN over the URL string, and a small CNN over inline images to
detect forged institutional marks.
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from typing import Any, Optional
from urllib.parse import urlsplit

if __package__ in (None, ""):  # pragma: no cover - script convenience only
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from mailguard.core.models import Attachment, ExtractedURL, ParsedEmail, SignalResult

SIGNAL_ID = "x7"
SIGNAL_NAME = "Attachment and URL"

# ----------------------------------------------------------------------
# Optional dependencies, all guarded, none required.
# ----------------------------------------------------------------------
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image

    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

# ======================================================================
# TRAINED MODEL SLOT 1 of 2: URL character level CNN
# ----------------------------------------------------------------------
# The model for this signal has NOT been trained yet.
# When it is, put the artefact path in MODEL_PATH below and the code
# will pick it up automatically on the next run. Until then the
# documented heuristic in _heuristic_score() runs instead, so the whole
# pipeline stays runnable end to end today.
#
# Expected artefact: a joblib dumped dict
#     {
#       "model": <callable scorer, or a torch nn.Module in eval mode>,
#       "vocab": {"a": 1, "b": 2, ...},   # optional, else URL_CHAR_VOCAB
#       "max_len": 200,                   # optional, else URL_MAX_LEN
#       "framework": "torch" | "sklearn"
#     }
# Input is the URL STRING, encoded character by character with
# encode_url() below into a fixed length integer sequence: a character
# CNN learns the token shapes that regexes here can only enumerate
# ("secure-", "-verify", "/login.php?id="). Output is a single
# P(malicious URL) in 0..1 per URL; the signal takes the maximum over
# every URL in the message, because one hostile link is enough.
# Expected feature order is defined by FEATURE_NAMES below. FEATURE_NAMES
# is the tabular contract for the heuristic, the evidence row, and
# ablation against the CNN; the CNN itself consumes the raw string.
# ======================================================================
MODEL_PATH: str = ""          # <-- trained model path goes here

URL_MAX_LEN: int = 200
URL_CHAR_VOCAB: str = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"

FEATURE_NAMES: list[str] = [
    # URL structure
    "url_count_norm",                  # how many links, normalised
    "punycode_url",                    # xn-- in a host
    "max_subdomain_depth_norm",        # deepest host, normalised
    "leet_substitution",               # digits standing in for letters
    "shortened_url",                   # link shortener in play
    "anchor_text_mismatch",            # visible text disagrees with href
    "link_domain_differs_from_sender",  # link host unrelated to From domain
    "ip_literal_host",                 # http://203.0.113.9/...
    "credential_path_keywords",        # /login, /verify, /secure, /kyc
    "at_sign_in_url",                  # userinfo@ trick
    "nonstandard_port",                # :8080, :8443
    "suspicious_tld",                  # .zip, .top, .xyz and friends
    "url_length_norm",                 # longest URL, normalised
    "redirect_chain_length_norm",      # only populated if resolution is on
    # Attachment risk
    "attachment_count_norm",
    "macro_office_document",           # OLE2 magic or vbaProject.bin
    "mime_magic_mismatch",             # declared type is not the real type
    "double_extension",                # invoice.pdf.exe
    "rtl_override_filename",           # U+202E filename reversal trick
    "executable_attachment",
    "archive_attachment",
    "attachment_size_norm",
    # Inline image forgery, scored by slot 2
    "logo_forgery_score",
]

# ======================================================================
# TRAINED MODEL SLOT 2 of 2: inline image logo CNN
# ----------------------------------------------------------------------
# The model for this signal has NOT been trained yet.
# When it is, put the artefact paths in LOGO_MODEL_PATH and
# LOGO_LABEL_MAP_PATH below and the code will pick them up automatically
# on the next run. Until then the documented heuristic in
# _logo_heuristic_score() runs instead.
#
# Expected artefacts:
#   LOGO_MODEL_PATH      a torch state dict (torch.save(model.state_dict()))
#                        for a small classifier over 64x64 RGB crops,
#                        loaded with LOGO_ARCH_FACTORY below;
#   LOGO_LABEL_MAP_PATH  a JSON map {"0": "none", "1": "hdfc", "2": "sbi",
#                        ...} matching the training label order.
# Input is each inline image decoded to RGB and resized to 64x64.
# Output is a brand label per image. The signal turns that into a score
# by asking whether the brand the image claims is a brand the SENDING
# DOMAIN is allowed to claim: a genuine HDFC mail carrying the HDFC mark
# is not evidence of anything, the same mark on mail from
# hdfc-verify.example is.
# Expected feature order is defined by LOGO_FEATURE_NAMES below.
# ======================================================================
LOGO_MODEL_PATH: str = ""      # <-- trained logo CNN state dict goes here
LOGO_LABEL_MAP_PATH: str = ""  # <-- trained logo label map JSON goes here
# Set this to a zero-argument callable returning the nn.Module whose
# state dict LOGO_MODEL_PATH holds. Kept as a slot rather than a hard
# coded architecture so the trainer owns the architecture, not this file.
LOGO_ARCH_FACTORY: Any = None

LOGO_FEATURE_NAMES: list[str] = [
    "inline_image_count_norm",   # how many inline images
    "logo_like_geometry",         # small, wide, banner-shaped
    "brand_claimed_in_text",      # body or subject names a protected brand
    "sender_may_claim_brand",     # the From domain is allowed that brand
    "total_image_area_norm",      # whole-body image mails hide text from filters
]

# ----------------------------------------------------------------------
# Redirect resolution, deliberately off.
# ----------------------------------------------------------------------
# Resolving a shortened link means this host issues an HTTP request to an
# address chosen by the attacker. That confirms delivery, leaks the fact
# and timing of analysis, exposes the analysis host's address, can be
# used as a tracking beacon, and in a targeted case is an attack surface
# aimed at the scanner itself. So it is off by default and an operator
# turns it on knowingly, ideally through an egress proxy.
RESOLVE_REDIRECTS: bool = os.environ.get("MAILGUARD_RESOLVE_REDIRECTS", "") == "1"
REDIRECT_TIMEOUT: float = 3.0
REDIRECT_MAX_HOPS: int = 5

# ----------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------
SHORTENER_DOMAINS: frozenset[str] = frozenset({
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rb.gy", "rebrand.ly", "tiny.cc", "shorturl.at", "s.id",
    "lnkd.in", "t.ly", "bl.ink", "snipurl.com", "shorte.st", "adf.ly",
    "trib.al", "mcaf.ee", "qr.ae", "u.to", "clck.ru", "v.gd", "soo.gd",
})

# TLDs that carry a disproportionate share of abuse, either because they
# are free or because they collide with file extensions (.zip, .mov).
SUSPICIOUS_TLDS: frozenset[str] = frozenset({
    "zip", "mov", "top", "xyz", "gq", "cf", "tk", "ml", "ga", "work",
    "click", "link", "country", "review", "loan", "kim", "men", "date",
    "stream", "quest", "cam", "fit", "rest", "monster", "cyou", "sbs",
    "buzz", "icu", "wang", "live", "surf", "bar", "casa", "beauty",
})

# Path and query tokens that say "this page wants a credential".
CREDENTIAL_PATH_TOKENS: tuple[str, ...] = (
    "login", "signin", "sign-in", "logon", "auth", "oauth", "sso",
    "verify", "verification", "validate", "confirm", "secure", "security",
    "account", "accounts", "update", "password", "passwd", "credential",
    "webmail", "owa", "portal", "wallet", "unlock", "recover", "reset",
    "kyc", "netbanking", "ebanking", "invoice", "payment", "billing",
    "session", "token", "redirect", "docusign", "onedrive", "sharepoint",
)

EXECUTABLE_EXTENSIONS: frozenset[str] = frozenset({
    "exe", "scr", "com", "pif", "bat", "cmd", "js", "jse", "vbs", "vbe",
    "wsf", "wsh", "hta", "msi", "msp", "cpl", "jar", "ps1", "psm1", "lnk",
    "reg", "dll", "sys", "app", "iso", "img", "vhd", "apk", "deb", "sh",
})

ARCHIVE_EXTENSIONS: frozenset[str] = frozenset({
    "zip", "rar", "7z", "tar", "gz", "bz2", "xz", "cab", "arj", "lzh", "ace",
})

MACRO_CAPABLE_EXTENSIONS: frozenset[str] = frozenset({
    "doc", "dot", "xls", "xlt", "xla", "ppt", "pot", "docm", "dotm",
    "xlsm", "xltm", "xlam", "pptm", "potm", "ppam", "sldm",
})

# Magic byte prefixes, longest first so PE/MZ does not shadow anything.
MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ole2"),   # legacy Office, msi, msg
    (b"\x50\x4b\x03\x04", "zip"),                     # zip, ooxml, jar, apk
    (b"\x25\x50\x44\x46", "pdf"),
    (b"\x7f\x45\x4c\x46", "elf"),
    (b"\x89\x50\x4e\x47", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x47\x49\x46\x38", "gif"),
    (b"\x52\x61\x72\x21", "rar"),
    (b"\x37\x7a\xbc\xaf", "7z"),
    (b"\x1f\x8b", "gzip"),
    (b"{\\rtf", "rtf"),
    (b"\x4c\x00\x00\x00\x01\x14\x02\x00", "lnk"),
    (b"MZ", "pe"),
)

# Declared MIME family against real magic family. Anything not listed
# here is not checked, because guessing generates false positives and a
# forensic tool should only assert what it can prove.
MIME_TO_MAGIC: dict[str, frozenset[str]] = {
    "application/pdf": frozenset({"pdf"}),
    "image/png": frozenset({"png"}),
    "image/jpeg": frozenset({"jpeg"}),
    "image/gif": frozenset({"gif"}),
    "application/zip": frozenset({"zip"}),
    "application/x-rar-compressed": frozenset({"rar"}),
    "application/msword": frozenset({"ole2", "rtf"}),
    "application/vnd.ms-excel": frozenset({"ole2", "zip"}),
    "application/vnd.ms-powerpoint": frozenset({"ole2"}),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": frozenset({"zip"}),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": frozenset({"zip"}),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": frozenset({"zip"}),
    "application/vnd.ms-excel.sheet.macroenabled.12": frozenset({"zip"}),
    "application/vnd.ms-word.document.macroenabled.12": frozenset({"zip"}),
    "text/plain": frozenset({"", "text"}),
    "application/rtf": frozenset({"rtf", "ole2"}),
}

# Brand tokens for the logo heuristic. Kept local rather than imported
# from x2 so that each signal module stays independently loadable; in a
# consolidated build both read one brand registry.
LOGO_BRAND_TOKENS: dict[str, tuple[str, ...]] = {
    "hdfc": ("hdfc.in", "hdfcbank.com"),
    "sbi": ("sbi.co.in", "onlinesbi.sbi"),
    "icici": ("icicibank.com",),
    "axis": ("axisbank.com",),
    "paypal": ("paypal.com",),
    "microsoft": ("microsoft.com", "office365.com", "outlook.com"),
    "office 365": ("microsoft.com", "office365.com"),
    "google": ("google.com", "gmail.com"),
    "amazon": ("amazon.in", "amazon.com"),
    "apple": ("apple.com", "icloud.com"),
    "docusign": ("docusign.net", "docusign.com"),
    "dhl": ("dhl.com",),
    "netflix": ("netflix.com",),
}

_LEET_IN_DOMAIN = re.compile(r"[a-z]+[0-9]+[a-z]+|[0-9]+[a-z]{3,}")
_IPV4_HOST = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_ANCHOR = re.compile(r"(?is)<a\b[^>]*href\s*=\s*[\"']?([^\"'>\s]+)[\"']?[^>]*>(.*?)</a>")
_TAGS = re.compile(r"(?s)<[^>]+>")
_DOMAIN_IN_TEXT = re.compile(r"\b(?:[a-z0-9][a-z0-9-]{0,62}\.)+[a-z]{2,24}\b", re.IGNORECASE)

_MODEL: Any = None
_MODEL_TRIED: bool = False
_LOGO_MODEL: Any = None
_LOGO_LABELS: Optional[dict[str, str]] = None
_LOGO_TRIED: bool = False


# ----------------------------------------------------------------------
# URL helpers
# ----------------------------------------------------------------------
def encode_url(url: str, max_len: int = URL_MAX_LEN, vocab: Optional[dict[str, int]] = None) -> list[int]:
    """Encode a URL as a fixed length integer sequence for the char CNN.

    Index 0 is the padding and out-of-vocabulary slot, so the model can
    learn that an unusual character is itself informative.
    """
    table = vocab or {ch: i + 1 for i, ch in enumerate(URL_CHAR_VOCAB)}
    encoded = [table.get(ch, 0) for ch in (url or "").lower()[:max_len]]
    return encoded + [0] * (max_len - len(encoded))


def _host_of(url: str) -> str:
    try:
        host = urlsplit(url if "://" in url else "http://" + url).hostname or ""
    except ValueError:
        return ""
    return host.lower().strip(".")


def _registrable_of(host: str) -> str:
    """Last two labels of a host. Deliberately naive and good enough for
    comparing a link host against a sending domain."""
    parts = [p for p in host.split(".") if p]
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def url_structure_risk(url: str, from_domain: str = "") -> tuple[float, dict[str, Any]]:
    """Structural risk of one URL, with the findings that produced it."""
    findings: dict[str, Any] = {
        "url": url,
        "host": "",
        "punycode": False,
        "subdomain_depth": 0,
        "leet": False,
        "ip_host": False,
        "at_sign": False,
        "nonstandard_port": False,
        "suspicious_tld": "",
        "credential_tokens": [],
        "length": len(url or ""),
        "unrelated_to_sender": False,
    }
    if not url:
        return 0.0, findings
    try:
        parts = urlsplit(url if "://" in url else "http://" + url)
    except ValueError:
        return 0.5, findings
    host = (parts.hostname or "").lower().strip(".")
    findings["host"] = host
    labels = [label for label in host.split(".") if label]

    findings["punycode"] = any(label.startswith("xn--") for label in labels)
    findings["subdomain_depth"] = max(len(labels) - 2, 0)
    findings["leet"] = bool(labels) and any(_LEET_IN_DOMAIN.search(label) for label in labels[:-1])
    findings["ip_host"] = bool(_IPV4_HOST.match(host))
    findings["at_sign"] = "@" in (parts.netloc or "")
    try:
        port = parts.port
    except ValueError:
        port = None
    findings["nonstandard_port"] = port is not None and port not in (80, 443)
    tld = labels[-1] if labels else ""
    findings["suspicious_tld"] = tld if tld in SUSPICIOUS_TLDS else ""
    haystack = f"{parts.path}?{parts.query}".lower()
    findings["credential_tokens"] = [t for t in CREDENTIAL_PATH_TOKENS if t in haystack]
    if from_domain and host:
        findings["unrelated_to_sender"] = _registrable_of(host) != _registrable_of(from_domain.lower())

    # Weighted structural risk. Punycode and a raw IP host are close to
    # dispositive on their own; depth, length and path tokens are only
    # weak evidence individually, which is why they are small terms.
    risk = 0.0
    risk += 0.45 if findings["punycode"] else 0.0
    risk += 0.40 if findings["ip_host"] else 0.0
    risk += 0.35 if findings["at_sign"] else 0.0
    risk += 0.20 if findings["leet"] else 0.0
    risk += 0.20 if findings["suspicious_tld"] else 0.0
    risk += 0.15 if findings["nonstandard_port"] else 0.0
    risk += min(findings["subdomain_depth"], 4) * 0.07
    risk += min(len(findings["credential_tokens"]), 3) * 0.08
    risk += 0.10 if findings["length"] > 120 else 0.0
    return max(0.0, min(1.0, risk)), findings


def resolve_redirect_chain(url: str) -> list[str]:
    """Follow a link and return the chain of URLs it walks through.

    Off unless RESOLVE_REDIRECTS is set, and wrapped so a network failure
    can never affect the verdict. See the comment on RESOLVE_REDIRECTS
    for why the default is off.
    """
    if not RESOLVE_REDIRECTS or not HAS_REQUESTS or not url:
        return []
    try:
        response = requests.get(
            url,
            allow_redirects=True,
            timeout=REDIRECT_TIMEOUT,
            stream=True,
            headers={"User-Agent": "MailGuard-AI/analysis"},
        )
        chain = [step.url for step in response.history[:REDIRECT_MAX_HOPS]]
        chain.append(response.url)
        response.close()
        return chain
    except Exception:
        return []


def _anchor_mismatches(html: str) -> list[tuple[str, str]]:
    """Anchors whose visible text names a different domain than the href.

    Only fires when the visible text actually contains a domain: link
    text reading "click here" is a different finding (x5's) and not a
    mismatch.
    """
    mismatches: list[tuple[str, str]] = []
    if not html:
        return mismatches
    for href, inner in _ANCHOR.findall(html):
        target_host = _host_of(href)
        if not target_host:
            continue
        visible = _TAGS.sub(" ", inner)
        for candidate in _DOMAIN_IN_TEXT.findall(visible):
            candidate_host = candidate.lower().strip(".")
            if candidate_host.rsplit(".", 1)[-1] in ("png", "jpg", "jpeg", "gif", "webp", "svg"):
                continue
            if _registrable_of(candidate_host) != _registrable_of(target_host):
                mismatches.append((candidate_host, target_host))
                break
    return mismatches


# ----------------------------------------------------------------------
# Attachment helpers
# ----------------------------------------------------------------------
def sniff_magic(raw: bytes) -> str:
    """Identify a file from its leading bytes, or return an empty string."""
    if not raw:
        return ""
    for prefix, name in MAGIC_SIGNATURES:
        if raw.startswith(prefix):
            return name
    return ""


def has_ooxml_macro(raw: bytes) -> bool:
    """True when an OOXML container carries a VBA project.

    A .xlsx is a zip; a .xlsm is the same zip with xl/vbaProject.bin
    inside it. Renaming the file changes nothing, which is exactly why
    the check is on the container and not the extension.
    """
    if not raw.startswith(b"\x50\x4b\x03\x04"):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            return any(
                name.lower().endswith("vbaproject.bin") for name in archive.namelist()
            )
    except (zipfile.BadZipFile, OSError, ValueError):
        return False


def _extensions(filename: str) -> list[str]:
    return [part.lower() for part in (filename or "").split(".")[1:] if part]


def attachment_risk(attachment: Attachment) -> tuple[float, dict[str, Any]]:
    """Risk of one attachment, with the findings that produced it."""
    findings: dict[str, Any] = {
        "filename": attachment.filename,
        "declared_type": attachment.content_type,
        "magic": attachment.magic_type or sniff_magic(attachment.raw),
        "macro": False,
        "macro_mechanism": "",
        "mime_mismatch": False,
        "double_extension": False,
        "rtl_override": False,
        "executable": False,
        "archive": False,
        "size_bytes": attachment.size_bytes,
    }
    extensions = _extensions(attachment.filename)
    last_extension = extensions[-1] if extensions else ""
    magic = findings["magic"]

    if magic == "ole2" and (last_extension in MACRO_CAPABLE_EXTENSIONS or not extensions):
        findings["macro"] = True
        findings["macro_mechanism"] = "OLE2 container, legacy Office macro format"
    elif has_ooxml_macro(attachment.raw):
        findings["macro"] = True
        findings["macro_mechanism"] = "vbaProject.bin present inside the OOXML zip"
    elif last_extension in MACRO_CAPABLE_EXTENSIONS and last_extension.endswith("m"):
        # Declared macro-enabled but the bytes were not available to
        # confirm it. Recorded as a weaker finding, and said so.
        findings["macro"] = True
        findings["macro_mechanism"] = f"macro-enabled extension .{last_extension}, bytes not verified"

    declared = (attachment.content_type or "").split(";")[0].strip().lower()
    expected = MIME_TO_MAGIC.get(declared)
    if expected and magic and magic not in expected:
        findings["mime_mismatch"] = True

    if len(extensions) >= 2 and last_extension in (EXECUTABLE_EXTENSIONS | ARCHIVE_EXTENSIONS):
        findings["double_extension"] = True
    findings["rtl_override"] = any(
        ch in (attachment.filename or "") for ch in ("‮", "‫", "‏")
    )
    findings["executable"] = last_extension in EXECUTABLE_EXTENSIONS or magic in ("pe", "elf", "lnk")
    findings["archive"] = last_extension in ARCHIVE_EXTENSIONS or magic in ("zip", "rar", "7z", "gzip")

    risk = 0.0
    risk += 0.55 if findings["macro"] else 0.0
    risk += 0.60 if findings["executable"] else 0.0
    risk += 0.50 if findings["double_extension"] else 0.0
    risk += 0.50 if findings["rtl_override"] else 0.0
    risk += 0.35 if findings["mime_mismatch"] else 0.0
    # An archive on its own is ordinary business traffic; it only matters
    # as a wrapper, so it is a small term.
    risk += 0.10 if findings["archive"] and not findings["executable"] else 0.0
    return max(0.0, min(1.0, risk)), findings


# ----------------------------------------------------------------------
# Inline image / logo forgery, trained model slot 2
# ----------------------------------------------------------------------
def _image_geometry(image: Attachment) -> Optional[tuple[int, int]]:
    """(width, height) of an inline image, or None if undeterminable."""
    if not HAS_PILLOW or not image.raw:
        return None
    try:
        with Image.open(io.BytesIO(image.raw)) as handle:
            return int(handle.width), int(handle.height)
    except Exception:
        return None


def _extract_logo_features(email: ParsedEmail) -> dict[str, float]:
    """Pull every feature in LOGO_FEATURE_NAMES out of the email."""
    images = email.inline_images or []
    text = f"{email.subject or ''} {email.text_body or ''}".lower()
    from_domain = (email.from_domain or "").lower()

    claimed = 0.0
    may_claim = 0.0
    for token, allowed in LOGO_BRAND_TOKENS.items():
        if token in text:
            claimed = 1.0
            if any(from_domain == d or from_domain.endswith("." + d) for d in allowed):
                may_claim = 1.0
            break

    logo_like = 0.0
    total_area = 0.0
    for image in images:
        geometry = _image_geometry(image)
        if geometry is None:
            # No Pillow, or an image we cannot decode. Fall back to size:
            # institutional marks are small files.
            if 0 < image.size_bytes <= 60_000:
                logo_like = max(logo_like, 0.5)
            continue
        width, height = geometry
        total_area += width * height
        if height and 1.2 <= width / height <= 8.0 and width <= 600 and height <= 200:
            logo_like = 1.0
    return {
        "inline_image_count_norm": min(len(images) / 4.0, 1.0),
        "logo_like_geometry": logo_like,
        "brand_claimed_in_text": claimed,
        "sender_may_claim_brand": may_claim,
        "total_image_area_norm": min(total_area / 500_000.0, 1.0) if total_area else 0.0,
    }


def _load_logo_model() -> Any:
    """Load the logo CNN once, or return None if the slot is empty."""
    global _LOGO_MODEL, _LOGO_LABELS, _LOGO_TRIED
    if _LOGO_TRIED:
        return _LOGO_MODEL
    _LOGO_TRIED = True
    if not (LOGO_MODEL_PATH and os.path.exists(LOGO_MODEL_PATH)):
        return None
    if not (HAS_TORCH and HAS_PILLOW and callable(LOGO_ARCH_FACTORY)):
        return None
    try:
        model = LOGO_ARCH_FACTORY()
        model.load_state_dict(torch.load(LOGO_MODEL_PATH, map_location="cpu"))
        model.eval()
        _LOGO_MODEL = model
        if LOGO_LABEL_MAP_PATH and os.path.exists(LOGO_LABEL_MAP_PATH):
            with open(LOGO_LABEL_MAP_PATH, "r", encoding="utf-8") as fh:
                _LOGO_LABELS = json.load(fh)
    except Exception:
        _LOGO_MODEL = None
        _LOGO_LABELS = None
    return _LOGO_MODEL


def _logo_model_score(email: ParsedEmail) -> tuple[float | None, str]:
    """Classify inline marks with the trained CNN, if the slot is filled.

    Returns (score, note). The score is not "is this a logo" but "is this
    a logo this sender has no right to display", which is the only form
    of the question that is evidence.
    """
    model = _load_logo_model()
    if model is None or not email.inline_images:
        return None, ""
    from_domain = (email.from_domain or "").lower()
    try:
        worst = 0.0
        note = ""
        for image in email.inline_images:
            if not image.raw:
                continue
            with Image.open(io.BytesIO(image.raw)) as handle:
                frame = handle.convert("RGB").resize((64, 64))
            pixels = [[list(frame.getpixel((x, y))) for x in range(64)] for y in range(64)]
            tensor = torch.tensor(pixels, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            with torch.no_grad():
                logits = model(tensor)[0]
            probabilities = torch.softmax(logits, dim=-1).tolist()
            best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            confidence = float(probabilities[best_index])
            label = (_LOGO_LABELS or {}).get(str(best_index), str(best_index)).lower()
            if label in ("none", "background", "unknown") or confidence < 0.6:
                continue
            allowed = LOGO_BRAND_TOKENS.get(label, ())
            entitled = any(from_domain == d or from_domain.endswith("." + d) for d in allowed)
            if not entitled and confidence > worst:
                worst = confidence
                note = f"{label} mark detected on mail from {from_domain or 'an unknown domain'}"
        return worst, note
    except Exception:
        return None, ""


def _logo_heuristic_score(features: dict[str, float]) -> tuple[float, str]:
    """Documented stand in until the logo CNN is trained.

    The CNN reads the pixels; this heuristic can only read the setting in
    which a forged mark appears, which is: the message names a protected
    brand in its text, it carries inline images shaped like an
    institutional mark, and the sending domain is not entitled to that
    brand. That combination scores 0.55, plus a little for a mail that is
    mostly image (a common way to keep text out of reach of filters).

    It says nothing about whether the image IS the brand's mark, so it
    cannot reach 1.0 and should not. Do not raise these numbers without
    the CNN: the honest ceiling of a scorer that has not looked at the
    image is low.
    """
    if features.get("sender_may_claim_brand", 0.0) > 0:
        return 0.0, ""
    if features.get("brand_claimed_in_text", 0.0) <= 0:
        return 0.0, ""
    score = 0.0
    note = ""
    if features.get("logo_like_geometry", 0.0) > 0 and features.get("inline_image_count_norm", 0.0) > 0:
        score = 0.55 * features["logo_like_geometry"]
        note = "brand named in text with logo-shaped inline image from an unentitled domain"
    score += 0.15 * features.get("total_image_area_norm", 0.0)
    return max(0.0, min(1.0, score)), note


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def _extract_details(email: ParsedEmail) -> dict[str, Any]:
    """Every feature in FEATURE_NAMES, plus analyst context."""
    from_domain = (email.from_domain or "").lower()
    urls: list[ExtractedURL] = list(email.urls or [])

    url_findings: list[dict[str, Any]] = []
    worst_url_risk = 0.0
    worst_url = ""
    punycode = 0.0
    max_depth = 0
    leet = 0.0
    ip_host = 0.0
    at_sign = 0.0
    port = 0.0
    bad_tld = 0.0
    credential_tokens = 0.0
    longest = 0
    shortened = 0.0
    unrelated = 0.0

    for entry in urls:
        target = entry.final_url or entry.url
        risk, findings = url_structure_risk(target, from_domain)
        findings["source"] = entry.source
        findings["declared_domain"] = entry.domain
        url_findings.append(findings)
        if risk > worst_url_risk:
            worst_url_risk, worst_url = risk, target
        punycode = max(punycode, 1.0 if findings["punycode"] else 0.0)
        max_depth = max(max_depth, findings["subdomain_depth"])
        leet = max(leet, 1.0 if findings["leet"] else 0.0)
        ip_host = max(ip_host, 1.0 if findings["ip_host"] else 0.0)
        at_sign = max(at_sign, 1.0 if findings["at_sign"] else 0.0)
        port = max(port, 1.0 if findings["nonstandard_port"] else 0.0)
        bad_tld = max(bad_tld, 1.0 if findings["suspicious_tld"] else 0.0)
        credential_tokens = max(credential_tokens, min(len(findings["credential_tokens"]) / 2.0, 1.0))
        longest = max(longest, findings["length"])
        host = findings["host"]
        if entry.is_shortened or _registrable_of(host) in SHORTENER_DOMAINS:
            shortened = 1.0
        if findings["unrelated_to_sender"]:
            unrelated = 1.0

    mismatches = _anchor_mismatches(email.html_body or "")

    # Redirect chains, only if an operator switched resolution on.
    chains: dict[str, list[str]] = {}
    if RESOLVE_REDIRECTS:
        for entry in urls:
            if entry.is_shortened or _registrable_of(_host_of(entry.url)) in SHORTENER_DOMAINS:
                chain = resolve_redirect_chain(entry.url)
                if chain:
                    chains[entry.url] = chain
    longest_chain = max((len(chain) for chain in chains.values()), default=0)

    attachment_findings: list[dict[str, Any]] = []
    macro = 0.0
    mime_mismatch = 0.0
    double_extension = 0.0
    rtl = 0.0
    executable = 0.0
    archive = 0.0
    largest = 0
    worst_attachment_risk = 0.0
    for attachment in email.attachments or []:
        risk, findings = attachment_risk(attachment)
        attachment_findings.append(findings)
        worst_attachment_risk = max(worst_attachment_risk, risk)
        macro = max(macro, 1.0 if findings["macro"] else 0.0)
        mime_mismatch = max(mime_mismatch, 1.0 if findings["mime_mismatch"] else 0.0)
        double_extension = max(double_extension, 1.0 if findings["double_extension"] else 0.0)
        rtl = max(rtl, 1.0 if findings["rtl_override"] else 0.0)
        executable = max(executable, 1.0 if findings["executable"] else 0.0)
        archive = max(archive, 1.0 if findings["archive"] else 0.0)
        largest = max(largest, int(findings["size_bytes"] or 0))

    logo_features = _extract_logo_features(email)
    logo_model_value, logo_model_note = _logo_model_score(email)
    if logo_model_value is not None:
        logo_score, logo_note = logo_model_value, logo_model_note
        logo_backed = True
    else:
        logo_score, logo_note = _logo_heuristic_score(logo_features)
        logo_backed = False

    return {
        "url_count_norm": min(len(urls) / 5.0, 1.0),
        "punycode_url": punycode,
        "max_subdomain_depth_norm": min(max_depth / 4.0, 1.0),
        "leet_substitution": leet,
        "shortened_url": shortened,
        "anchor_text_mismatch": 1.0 if mismatches else 0.0,
        "link_domain_differs_from_sender": unrelated,
        "ip_literal_host": ip_host,
        "credential_path_keywords": credential_tokens,
        "at_sign_in_url": at_sign,
        "nonstandard_port": port,
        "suspicious_tld": bad_tld,
        "url_length_norm": min(longest / 200.0, 1.0),
        "redirect_chain_length_norm": min(longest_chain / 5.0, 1.0),
        "attachment_count_norm": min(len(email.attachments or []) / 3.0, 1.0),
        "macro_office_document": macro,
        "mime_magic_mismatch": mime_mismatch,
        "double_extension": double_extension,
        "rtl_override_filename": rtl,
        "executable_attachment": executable,
        "archive_attachment": archive,
        "attachment_size_norm": min(largest / 5_000_000.0, 1.0),
        "logo_forgery_score": logo_score,
        "_url_findings": url_findings,
        "_attachment_findings": attachment_findings,
        "_anchor_mismatches": mismatches,
        "_worst_url": worst_url,
        "_worst_url_risk": worst_url_risk,
        "_worst_attachment_risk": worst_attachment_risk,
        "_redirect_chains": chains,
        "_logo_features": logo_features,
        "_logo_note": logo_note,
        "_logo_model_backed": logo_backed,
    }


def _extract_features(email: ParsedEmail) -> dict[str, float]:
    """Pull every feature in FEATURE_NAMES out of the email."""
    details = _extract_details(email)
    return {name: float(details[name]) for name in FEATURE_NAMES}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def _heuristic_score(features: dict[str, float]) -> float:
    """Documented stand in until the URL CNN is trained.

    Three parts, combined by a soft maximum rather than a sum:

      url_part        = the worst single URL's structural risk, plus
                        0.30 for an anchor text mismatch and 0.15 for a
                        shortened link, because those two are properties
                        of the message rather than of one URL;
      attachment_part = the worst single attachment's risk;
      logo_part       = slot 2's score, capped at 0.55 while it is a
                        heuristic.

    The three are combined as max(parts) + 0.35 * (sum of the rest),
    which keeps a single decisive finding (a macro document, a punycode
    host) near its own value while letting several independent weak
    findings add up. A plain sum would let five cosmetic URL quirks
    outscore a live executable, and a plain maximum would ignore the
    shape of a mail that is weakly wrong in every direction at once.

    Whoever trains the character CNN should beat this on URLs
    specifically: everything here is an enumerated pattern, so it is
    blind to any trick nobody has written down yet, which is precisely
    what a learned character model is for.
    """
    url_part = 0.0
    url_part = max(url_part, _url_part_from_features(features))
    attachment_part = _attachment_part_from_features(features)
    logo_part = min(features.get("logo_forgery_score", 0.0), 1.0)

    parts = sorted([url_part, attachment_part, logo_part], reverse=True)
    score = parts[0] + 0.35 * sum(parts[1:])
    return max(0.0, min(1.0, score))


def _url_part_from_features(features: dict[str, float]) -> float:
    """URL half of the heuristic, rebuilt from the flattened features."""
    risk = 0.0
    risk += 0.45 * features.get("punycode_url", 0.0)
    risk += 0.40 * features.get("ip_literal_host", 0.0)
    risk += 0.35 * features.get("at_sign_in_url", 0.0)
    risk += 0.30 * features.get("anchor_text_mismatch", 0.0)
    risk += 0.20 * features.get("leet_substitution", 0.0)
    risk += 0.20 * features.get("suspicious_tld", 0.0)
    risk += 0.15 * features.get("shortened_url", 0.0)
    risk += 0.15 * features.get("nonstandard_port", 0.0)
    risk += 0.16 * features.get("credential_path_keywords", 0.0)
    risk += 0.14 * features.get("max_subdomain_depth_norm", 0.0)
    risk += 0.10 * features.get("link_domain_differs_from_sender", 0.0)
    risk += 0.08 * features.get("url_length_norm", 0.0)
    risk += 0.10 * features.get("redirect_chain_length_norm", 0.0)
    return min(risk, 1.0)


def _attachment_part_from_features(features: dict[str, float]) -> float:
    """Attachment half of the heuristic, rebuilt from the features."""
    risk = 0.0
    risk += 0.60 * features.get("executable_attachment", 0.0)
    risk += 0.55 * features.get("macro_office_document", 0.0)
    risk += 0.50 * features.get("double_extension", 0.0)
    risk += 0.50 * features.get("rtl_override_filename", 0.0)
    risk += 0.35 * features.get("mime_magic_mismatch", 0.0)
    risk += 0.10 * features.get("archive_attachment", 0.0)
    return min(risk, 1.0)


def _load_model() -> Any:
    """Load the trained URL model once, or return None if the slot is empty."""
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


def _url_model_score(urls: list[str]) -> float | None:
    """Worst P(malicious) over the message's URLs from the char CNN."""
    artefact = _load_model()
    if artefact is None or not urls:
        return None
    try:
        if isinstance(artefact, dict):
            model = artefact.get("model")
            vocab = artefact.get("vocab")
            max_len = int(artefact.get("max_len", URL_MAX_LEN))
            framework = artefact.get("framework", "torch")
        else:
            model, vocab, max_len, framework = artefact, None, URL_MAX_LEN, "sklearn"
        if model is None:
            return None
        encoded = [encode_url(url, max_len=max_len, vocab=vocab) for url in urls]
        if framework == "torch":
            if not HAS_TORCH:
                return None
            with torch.no_grad():
                logits = model(torch.tensor(encoded, dtype=torch.long))
            values = torch.sigmoid(logits.reshape(len(encoded), -1)[:, -1]).tolist()
        elif hasattr(model, "predict_proba"):
            values = [float(row[1]) for row in model.predict_proba(encoded)]
        else:
            values = [float(v) for v in model.predict(encoded)]
        return max(0.0, min(1.0, max(values)))
    except Exception:
        return None


def _model_score(features: dict[str, float], urls: Optional[list[str]] = None) -> float | None:
    """Score with the trained models if a slot is filled, else None.

    The URL CNN replaces the URL half of the heuristic and is combined
    with the attachment half and the logo score exactly as the heuristic
    combines them, so switching the slot on changes one term rather than
    the meaning of the whole signal. The `urls` argument is a documented
    departure from the other signals for the same reason as in x5: a
    character CNN consumes strings, not tabular features.
    """
    url_value = _url_model_score(urls or [])
    if url_value is None:
        return None
    attachment_part = _attachment_part_from_features(features)
    logo_part = min(features.get("logo_forgery_score", 0.0), 1.0)
    parts = sorted([url_value, attachment_part, logo_part], reverse=True)
    return max(0.0, min(1.0, parts[0] + 0.35 * sum(parts[1:])))


def _evidence_row(details: dict[str, Any]) -> str:
    """One short printable line an analyst reads."""
    parts: list[str] = []
    for findings in details["_attachment_findings"]:
        if findings["macro"]:
            parts.append(f"{findings['filename']}: macro document ({findings['macro_mechanism']})")
        elif findings["executable"]:
            parts.append(f"{findings['filename']}: executable payload")
        elif findings["double_extension"]:
            parts.append(f"{findings['filename']}: double extension")
        elif findings["rtl_override"]:
            parts.append(f"{findings['filename']}: right-to-left override in filename")
        elif findings["mime_mismatch"]:
            parts.append(
                f"{findings['filename']}: declared {findings['declared_type']} but bytes are "
                f"{findings['magic'] or 'unknown'}"
            )
    if details["_anchor_mismatches"]:
        shown, target = details["_anchor_mismatches"][0]
        parts.append(f"link text says {shown} but points to {target}")
    worst = details["_worst_url"]
    if worst and details["_worst_url_risk"] >= 0.3:
        host = _host_of(worst)
        reasons: list[str] = []
        if details["punycode_url"]:
            reasons.append("punycode")
        if details["ip_literal_host"]:
            reasons.append("raw IP host")
        if details["suspicious_tld"]:
            reasons.append("abuse-prone TLD")
        if details["credential_path_keywords"]:
            reasons.append("credential path")
        if details["max_subdomain_depth_norm"] >= 0.75:
            reasons.append("deep subdomain")
        parts.append(f"link {host}" + (f" ({', '.join(reasons)})" if reasons else ""))
    if details["shortened_url"]:
        chains = details["_redirect_chains"]
        if chains:
            first = next(iter(chains.values()))
            parts.append(f"shortened link resolving to {_host_of(first[-1])}")
        else:
            parts.append("shortened link, not resolved")
    if details["_logo_note"]:
        parts.append(details["_logo_note"])
    if not parts:
        counts = (
            f"{len(details['_url_findings'])} links, "
            f"{len(details['_attachment_findings'])} attachments"
        )
        return f"no structural risk found in {counts}"
    row = "; ".join(parts[:4])
    return row if len(row) <= 220 else row[:217] + "..."


def run(email: ParsedEmail) -> SignalResult:
    """Score the message's links and attachments. Never raises."""
    try:
        if not (email.urls or email.attachments or email.inline_images):
            return SignalResult(
                signal_id=SIGNAL_ID,
                name=SIGNAL_NAME,
                score=0.0,
                status="abstain",
                evidence_row="no links, attachments or inline images to inspect",
                details={"reason": "no payload present"},
            )
        details = _extract_details(email)
        features = {name: float(details[name]) for name in FEATURE_NAMES}
        urls = [entry.final_url or entry.url for entry in (email.urls or [])]
        model_value = _model_score(features, urls)
        score = model_value if model_value is not None else _heuristic_score(features)
        public: dict[str, Any] = {k: v for k, v in details.items() if not k.startswith("_")}
        public.update(
            url_findings=details["_url_findings"],
            attachment_findings=[
                {k: v for k, v in findings.items()} for findings in details["_attachment_findings"]
            ],
            anchor_mismatches=details["_anchor_mismatches"],
            redirect_chains=details["_redirect_chains"],
            redirects_resolved=RESOLVE_REDIRECTS,
            logo_features=details["_logo_features"],
            logo_model_backed=details["_logo_model_backed"],
            url_domains=sorted(
                {findings["host"] for findings in details["_url_findings"] if findings["host"]}
            ),
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
            evidence_row=f"attachment and URL signal could not run: {type(exc).__name__}",
            details={"error": repr(exc)},
        )
