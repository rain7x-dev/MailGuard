# MailGuard AI

MailGuard AI decides whether an email is fraudulent, works out how far its
sender can honestly be traced, and turns both into evidence an investigator
can use. Every incoming message is scored by seven independent signals, each
returning a score between 0 and 1 plus one printable line of evidence. An
explainable fusion model turns the seven scores into a probability of fraud
and one of three actions: BLOCK, WARN or PASS. Underneath every verdict sits
a forensic trace of the message's route that believes only what our own
servers wrote, and says plainly where the trail stops.

## The seven signals

| ID | Name | Owner half | Model backed |
| --- | --- | --- | --- |
| x1 | Authentication (SPF, DKIM, DMARC, alignment) | forensics | no, rule based |
| x2 | Identity (cousin domains, display name, Reply-To) | ML | yes, slot empty, heuristic today |
| x3 | Infrastructure (domain age, registrar, MX, NS, IP reputation) | forensics | yes, slot empty, heuristic today |
| x4 | Sender Baseline (does this look like this sender?) | ML | yes, slot empty, heuristic today |
| x5 | Intent (what is the message asking for?) | ML | yes, slot empty, heuristic today |
| x6 | Origin and Geo (attribution tier, timezone, datacentre) | forensics | no, rule based |
| x7 | Attachment and URL (payload, link structure, logos) | ML | yes, slots empty, heuristic today |

A signal that cannot run returns `status="abstain"`. That means missing
data, not a clean score, and fusion treats it that way.

## Pipeline

```
   .eml file         IMAP mailbox         stdin (milter)
       \                  |                   /
        +---------- INGEST (hash first) -----+
                          |
                        PARSE     headers, MIME, URLs (text, HTML, QR, OCR)
                          |
                  TRUST BOUNDARY  which Received hops our servers wrote
                          |
                   ATTRIBUTION    tier 1 / 2 / 3 from the boundary IP
                          |
     +------+------+------+------+------+------+
     |      |      |      |      |      |      |       seven signals,
     x1     x2     x3     x4     x5     x6     x7      in parallel, each
     |      |      |      |      |      |      |       inside a time budget
     +------+------+------+------+------+------+
                          |
                        FUSION    explainable additive model
                          |
                       VERDICT    BLOCK / WARN / PASS + contributions
                          |
          +---------------+----------------+
          |               |                |
    CAMPAIGN GRAPH  EVIDENCE LEDGER    PDF REPORT
```

## The trust boundary

Every mail server that handles a message prepends a `Received:` header. Read
bottom to top they look like a complete travel history. They are not. A
`Received` header is just text: any server on the path can write anything,
and an attacker sending from their own machine can fabricate five plausible
hops before the message ever leaves.

The only headers we can trust are the ones written by servers we control.
So MailGuard starts at the top of the chain (index 0, added by our own mail
server) and walks down. A hop is trusted while its `by` host matches
`TRUSTED_HOSTS`. The first hop that does not match is the trust boundary;
everything from there down is attacker writable and reported as
unverifiable, never believed. The walk also stops where one of our servers
recorded an external peer, so a fabricated hop that claims `by
gw.example.org` below that point is caught as a forgery rather than
trusted. The origin candidate is the peer IP our last trusted server
recorded, the earliest address we can actually stand behind.

Worked example (`samples/forged_headers.eml`, trusted hosts
`mx.example.org`, `gw.example.org`):

```
[0] trusted    by mx.example.org          from gw.example.org [198.51.100.77]
---- trust boundary: everything below this line is attacker writable ----
[1] UNVERIFIED by gw.example.org          from mail.rt-business.example [192.0.2.10]
[2] UNVERIFIED by mail.rt-business.example from relay2.moscow-telecom.example [192.0.2.44]
[3] UNVERIFIED by relay2.moscow-telecom.example from office-pc [192.0.2.200]
```

Hop 0 is real: our MX wrote it. It records a connection from 198.51.100.77,
which has no reverse DNS and only introduced itself as `gw.example.org`
(anyone can type that). Our gateway never touched this message, so hop 1,
which claims `by gw.example.org`, was written by the sender. Hops 1 to 3
describe a journey through Russia that never happened. MailGuard reports
198.51.100.77 as the origin candidate, lists the three hops as claims, and
flags hop 1 as a forgery.

Attribution then takes that one IP and assigns a tier:

| Tier | Name | Meaning |
| --- | --- | --- |
| 1 | `direct_ip` | the sender's own network connected to us; ASN, country, ISP describe the origin |
| 2 | `provider_bounded` | webmail, ESP or hosting; only the provider can identify the account holder |
| 3 | `anonymised` | VPN, proxy, Tor, or the chain could not be walked; the concealment is the finding |

## Install and run

Python 3.11 or newer. Nothing beyond the standard library is required.

```bash
pip install -r requirements-forensics.txt   # optional extras, forensics half
pip install -r requirements-ml.txt          # optional extras, ML half

python -m mailguard.cli --eml samples/cousin_domain_phish.eml
python -m mailguard.cli --eml samples/forged_headers.eml --json
python -m mailguard.cli --stdin < samples/legitimate.eml
python -m mailguard.cli --eml samples/cousin_domain_phish.eml --report case.pdf
MAILGUARD_IMAP_PASSWORD=... python -m mailguard.cli --imap-host imap.example.org --imap-user analyst --imap-limit 5
```

Options: `--trusted-host HOST` (repeatable, replaces the default list),
`--offline` (no DNS, RDAP or AbuseIPDB lookups), `--timeout-ms N` (per
signal budget, default 400), `-v` (show why a signal module was skipped).

Configuration lives in `mailguard/core/config.py`, and every value can be
overridden by environment variable: `MAILGUARD_TRUSTED_HOSTS`,
`MAILGUARD_GEOLITE_DB`, `MAILGUARD_GEOLITE_ASN_DB`, `MAILGUARD_ABUSEIPDB_KEY`,
`MAILGUARD_SIGNAL_TIMEOUT_MS`, `MAILGUARD_NETWORK=0`, and others listed in
`load_config()`. **Set `TRUSTED_HOSTS` to your own mail servers before
relying on any result.**

The CLI runs with whatever exists: missing signal modules are skipped and
named, and without the fusion module it prints the signal table with a line
saying fusion is not available yet.

## Trained model slots

Every slot below ships empty. Each file has a `MODEL_PATH` constant; put the
artefact path there and the next run picks it up. Until then a documented
heuristic runs, so the pipeline works end to end today.

| Signal / stage | File | Constant | Expected artefact |
| --- | --- | --- | --- |
| x2 Identity | `mailguard/signals/x2_identity.py` | `MODEL_PATH` | joblib binary classifier with `predict_proba`, features in `FEATURE_NAMES` order |
| x3 Infrastructure | `mailguard/signals/x3_infrastructure.py` | `MODEL_PATH` | joblib gradient boosted classifier with `predict_proba(X) -> (n, 2)`, 9 features in `FEATURE_NAMES` order |
| x4 Sender Baseline | `mailguard/signals/x4_sender_baseline.py` | `MODEL_PATH` | joblib dict with a fitted IsolationForest and its feature names |
| x5 Intent | `mailguard/signals/x5_intent.py` | `MODEL_PATH` | directory from `save_pretrained()`: fine tuned DistilBERT sequence classifier |
| x7 URL | `mailguard/signals/x7_attachment_url.py` | `MODEL_PATH` | joblib dict with a URL character model and its vocabulary |
| x7 Logo | `mailguard/signals/x7_attachment_url.py` | `LOGO_MODEL_PATH` | torch state_dict for a 64x64 logo classifier plus a JSON label map |
| Fusion | `mailguard/fusion/ebm_fusion.py` | `MODEL_PATH` | joblib `ExplainableBoostingClassifier` over 14 columns, supporting `explain_local()` |
| Calibration | `mailguard/fusion/ebm_fusion.py` | `CALIBRATOR_PATH` | joblib isotonic or Platt calibrator |

x1 and x6 are rule based by design and have no slot. The ML half's slots
are described in full in `README_ML.md`.

## Project split

**Forensics half:** `mailguard/__init__.py`, `mailguard/core/config.py`,
`mailguard/ingest/`, `mailguard/parsing/`, `mailguard/forensics/`,
`mailguard/signals/__init__.py` (signal auto discovery),
`mailguard/signals/x1_authentication.py`,
`mailguard/signals/x3_infrastructure.py`, `mailguard/signals/x6_origin_geo.py`,
`mailguard/cli.py`, `samples/`, `requirements-forensics.txt`, this README.

**ML half:** `mailguard/signals/x2_identity.py`,
`mailguard/signals/x4_sender_baseline.py`, `mailguard/signals/x5_intent.py`,
`mailguard/signals/x7_attachment_url.py`, `mailguard/fusion/`,
`mailguard/graph/`, `mailguard/evidence/`, `tests/`, `requirements-ml.txt`,
`README_ML.md`.

**Shared contract:** `mailguard/core/models.py` defines every object the two
halves pass between them. It must not be changed unilaterally: any change is
agreed by both halves and made identically on both sides.

Signals are discovered automatically (`discover_signals()` imports every
`x<N>_*.py` module in `mailguard/signals/`), so neither half edits a shared
registry to add a signal.
