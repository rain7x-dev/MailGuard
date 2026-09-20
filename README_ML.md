# MailGuard AI - machine learning and output half

This document covers the half of MailGuard that scores four of the seven
signals, fuses all seven into a verdict, links messages into campaigns, keeps
the tamper evident case file, and produces the forensic report.

The other half (ingest, MIME and header parsing, the trust boundary walk,
origin attribution, signals x1, x3 and x6, and the CLI) is built
independently. The two halves meet at `mailguard/core/models.py`, which both
sides hold byte for byte identical. Everything here takes a `ParsedEmail` and
works whether or not the other half has run.

```
mailguard/
  core/models.py                shared data contract (identical in both halves)
  signals/x2_identity.py        x2  is the sender who they claim to be?
  signals/x4_sender_baseline.py x4  does this message look like this sender?
  signals/x5_intent.py          x5  what is the message asking for?
  signals/x7_attachment_url.py  x7  what is the payload, where does it point?
  fusion/ebm_fusion.py          seven scores -> one explainable probability
  fusion/verdict.py             probability -> five classes, three actions
  graph/campaign_graph.py       artefact graph, campaign resolution
  evidence/ledger.py            hash chained, append only case file
  evidence/pdf_report.py        the document an investigator receives
tests/factory.py                fully populated fake objects
tests/test_fusion.py            plain assert tests, no pytest needed
```

Nothing in this half requires a single third party package to import or run.
Every optional dependency is guarded by `try/except ImportError` with a
documented fallback. See `requirements-ml.txt`.

---

## 1. The four signals

Each signal returns a score in `[0, 1]`, a status, and one printable line of
evidence. A signal never raises: on failure it returns `status="abstain"`,
`score=0.0` and an evidence row saying why.

### x2 Identity - carries most business email compromise

BEC mail passes SPF, DKIM and DMARC, because the attacker really does own
`hdfc-verify.example` and really did sign for it. What they cannot do is make
that domain be `hdfc.in`. x2 is where that gap is measured, and it is the most
thorough signal in this half for that reason.

| What it computes | How |
| --- | --- |
| Cousin domain proximity | Unicode confusable folding (a hand written subset of the Unicode confusables table, plus digit-for-letter substitution) combined with Levenshtein distance, both pure Python. Six separate registration tricks are scored separately: homoglyph (0.99), protected domain as a subdomain of something else (0.97), one character edit (0.93), brand label embedded in a decorated label (0.90), transposed label (0.88), TLD swap (0.85). |
| Display name mismatch | `"HDFC Bank" <accounts@hdfc-verify.example>`: the display name claims a brand the sending domain is not on the allow list for. |
| Reply-To divergence | Reply-To differing from From, scored higher when it lands at free mail. |
| First contact | Whether this address has written to this recipient before, read from a JSON history store (a database table in production). |
| Thread hijack | `In-Reply-To` referencing a Message-ID this installation has never seen. Forging a reply to a conversation that never happened costs an attacker nothing; a real thread hijack requires mailbox access. |

Fifteen features, listed in `FEATURE_NAMES`. Being the genuine protected domain
subtracts from the score, the only negative term in the signal.

### x4 Sender Baseline - the only signal that catches a compromised account

When a real account is taken over, every other signal is clean **by
definition**: real domain, passing authentication, correct display name, real
thread, old and trusted sender. The only thing that changed is who is typing.
x4 compares the message against what the address has done before:

- **Embedding drift** from the mean of that sender's previous bodies.
  Three backends, tried in order: `sentence-transformers`, then scikit-learn
  TF-IDF cosine distance, then a pure Python character 4-gram Jaccard
  distance. The evidence row names which one was used, because evidence that
  does not say how it was measured is not evidence.
- **Send hour deviation** using circular statistics, because hours wrap at
  midnight and a linear mean of 23:00 and 01:00 is noon.
- **Volume anomaly** against that sender's normal daily rate.

**x4 abstains below five prior messages.** See section 4.

### x5 Intent - carries commodity phishing

Ignores the sender entirely and reads the request. Seven phrase families,
defined as module level constants so an analyst can read exactly what fired:
urgency, explicit deadlines, payment redirection, invoice manipulation,
credential harvesting, bank detail change requests, secrecy and authority
pressure, consequence threats. Subject urgency is scored separately from body
urgency, because the subject is where the attacker spends their attention.

A bank detail change request is scored on its own line, because that single
sentence is the payload of nearly every successful BEC.

### x7 Attachment and URL

- **URL structure**: punycode hosts, subdomain depth, digits standing in for
  letters, shorteners, anchor text disagreeing with its href, link domains
  unrelated to the sending domain, raw IP hosts, `userinfo@` tricks, non
  standard ports, credential flavoured paths, abuse prone TLDs.
- **Redirect resolution**: implemented, and **off by default**
  (`RESOLVE_REDIRECTS`). Fetching an attacker supplied URL from the analysis
  host confirms delivery, leaks the timing of analysis, exposes the scanner's
  address, and in a targeted case is an attack surface aimed at the scanner.
  An operator turns it on knowingly, ideally through an egress proxy.
- **Attachment risk**: macro bearing Office documents detected two ways, the
  OLE2 magic `\xd0\xcf\x11\xe0` for legacy binary formats and the presence of
  `vbaProject.bin` inside the OOXML zip; MIME type against actual magic bytes;
  double extensions; right-to-left override filenames; executables; archives.

---

## 2. Fusion: why this is not a neural network

```
P(fraud) = sigmoid( f0 + sum_i f_i(x_i) + sum_ij f_ij(x_i, x_j) )
```

The fusion model is an Explainable Boosting Machine: a generalised additive
model with pairwise interaction terms. Each `f_i` is a learned shape function
over one signal's score, so the contribution of every signal is **separable
and printable**.

In plain English: the model is not allowed to mix the signals into one
inseparable calculation. It has to form an opinion about each signal on its
own, then a small number of opinions about named pairs, and add them up. So
the final score can always be written as a list, and the list is the evidence
table in the report:

```
x2  Identity                0.91   +3.28 points   cousin domain of hdfc.in, ...
x5  Intent                  0.86   +2.75 points   bank detail change: all future ...
x7  Attachment and URL      0.78   +2.65 points   macro document (vbaProject.bin ...)
x4  Sender Baseline       abstain   +0.60 points   no baseline: 0 prior messages
...
interaction x2 with x5             +1.72 points
base rate (intercept)              -4.20 points
                                  ---------------
total log-odds                     +9.27  ->  p = 0.9999  ->  BLOCK
```

A deep network over the same seven inputs would score marginally better on a
held out set and would be impossible to decompose like this. **A verdict you
cannot explain is not evidence**, and evidence is the entire product. That
trade is made once, deliberately, in `fusion/ebm_fusion.py`.

Two consequences are enforced in code:

- `contributions` sums **exactly** to the logit, including the intercept and
  every active interaction term, so an analyst can check the arithmetic of the
  verdict by adding up the table. `tests/test_fusion.py` asserts this.
- If the trained EBM produces a probability but its local explanation cannot
  be read, the module **refuses to use that probability** and falls back to the
  documented heuristic for both the score and the attribution. Printing a
  probability with an attribution that does not reconstruct it would be worse
  than having no model at all.

### Feature vector

`SIGNAL_ORDER = ["x1","x2","x3","x4","x5","x6","x7"]` and the vector is
**fourteen** values: seven scores, then seven `is_present` flags. The model
learns what an absence means from the flags, instead of this code asserting
that absence means nothing.

### Calibration

`calibrate(raw_score)` has a slot for an isotonic or Platt calibrator and
defaults to the identity function. Calibration is what makes 0.9 mean roughly
nine in ten rather than an arbitrary ordering. Since the three action bands are
defined as probability cut points, an uncalibrated score would make the whole
policy meaningless.

A property worth knowing: with the placeholder priors, **no single signal at
the top of its range reaches BLOCK on its own**. A maxed out lone signal is
indistinguishable from a broken signal, so BLOCK requires corroboration. A
homoglyph domain with every other signal measured clean scores `p = 0.35`, and
in practice `p = 0.49` (WARN), because a newly registered domain also has no
sender baseline, so x4 abstains and adds its missingness term. Either way the
mail reaches the user and the analyst sees it.

---

## 3. Abstain is not zero

| Status | Means | Effect on fusion |
| --- | --- | --- |
| `ok`, score `0.0` | The signal ran and found nothing suspicious. | Score 0.0, `is_present = 1`. Contributes **0.00 points**. |
| `abstain` | The signal **could not run**. | Score 0.0, `is_present = 0`. Contributes its **missingness prior** (0.25 to 0.60 points). |

Why it matters: **a brand new sender with no history is exactly the profile an
attacker presents.** A freshly registered cousin domain has no sender baseline
by construction, no prior thread, and no correspondence record. Scoring that
as a clean neutral 0.0 would hand the attacker a better score than a genuine
correspondent with an unusual message, which is precisely backwards.

Equally, absence is only *mildly* adverse. All seven signals abstaining sits at
about `p = 0.18`, which is PASS. Being unable to measure anything is not
evidence of fraud; it is just not evidence of innocence either.

In the report, an abstaining signal is printed as `abstain, not scored` on a
shaded row, never as `0.00`, so no reader mistakes missing data for a pass.

---

## 4. Three actions, and the arithmetic behind them

| Action | Band | What happens |
| --- | --- | --- |
| `PASS` | `p < 0.35` | Delivered normally. |
| `WARN` | `0.35 <= p < 0.80` | Delivered with a banner, queued for an analyst, releasable by the user. |
| `BLOCK` | `p >= 0.80` | Quarantined, analyst release only. |

**The false positive arithmetic.** Take 100,000 messages a day, a mid-sized
enterprise. A detector at a 0.1 percent false positive rate is doing well by
any published benchmark. 0.1 percent of 100,000 is **100 legitimate messages
affected every day** - roughly 3,000 a month of invoices, customer replies and
job applications.

A system that hard blocks all 100 of them is switched off within a fortnight,
and then it catches nothing at all. That is why the middle band exists and why
WARN is the default for uncertainty: the user keeps their mail, the analyst
gets a queue ordered by probability, and the operator keeps the tool.

The five verdict classes give the analyst the shape of the attack, not just its
strength, because Impersonated and Phishing go to different playbooks:

| Class | When |
| --- | --- |
| `Legitimate` | `p < 0.35` |
| `Suspicious` | `0.35 <= p < 0.60`, or higher with neither carrier signal high |
| `Impersonated` | `p >= 0.60` and identity (x2) high, intent not |
| `Phishing` | `p >= 0.60` and intent (x5) high, identity not |
| `Fraud` | `p >= 0.60` and both high: a forged identity making a fraudulent request is the complete attack |

An abstaining x2 or x5 is never read as a low score when choosing the class.

---

## 5. Trained model slots

None of these models are trained yet. Every slot ships with the same
convention: set `MODEL_PATH`, and the code picks the artefact up on the next
run with no other change. Until then the documented heuristic runs, so the
pipeline works end to end today.

| Signal / stage | File | Constant | Expected artefact | Expected input | Heuristic fallback today |
| --- | --- | --- | --- | --- | --- |
| x2 Identity | `signals/x2_identity.py` | `MODEL_PATH` | joblib dumped binary classifier with `predict_proba(X) -> (n, 2)`; positive class = "presented identity is forged" | 15 tabular features in `FEATURE_NAMES` order | Weighted sum of structural findings: cousin proximity 0.55, thread hijack 0.35, display name mismatch 0.20, Reply-To free mail 0.15, first contact 0.08, homoglyph bonus 0.10, genuine protected domain -0.25 |
| x4 Sender Baseline | `signals/x4_sender_baseline.py` | `MODEL_PATH` | joblib dumped dict `{"model": fitted IsolationForest, "embedding_model": name, "feature_names": [...], "score_scale": [lo, hi]}`; fit on known-good mail only | 8 tabular features in `FEATURE_NAMES` order | `0.55*drift + 0.25*hour deviation + 0.20*volume anomaly`, plus small terms for a new recipient, off-hours send and a dormant account. Abstains below 5 prior messages |
| x5 Intent | `signals/x5_intent.py` | `MODEL_PATH` | **directory** from `save_pretrained()`: fine tuned DistilBERT sequence classifier, loadable by `AutoTokenizer` / `AutoModelForSequenceClassification`; fraud label named in `config.id2label` | subject and body joined by a newline, truncated to `MAX_TOKENS` (512) | Weighted phrase family scorer: bank detail change 0.42, credential harvest 0.36, payment redirection 0.30, login CTA 0.22, invoice 0.18, pressure families 0.10-0.14, plus an action-times-urgency interaction of 0.15 |
| x7 URL | `signals/x7_attachment_url.py` | `MODEL_PATH` | joblib dumped dict `{"model": torch module or sklearn scorer, "vocab": {...}, "max_len": 200, "framework": "torch"\|"sklearn"}` | the URL **string**, encoded by `encode_url()` into a fixed length integer sequence; index 0 is padding and out of vocabulary | Enumerated structural risk per URL (punycode 0.45, IP host 0.40, `@` 0.35, anchor mismatch 0.30, leet 0.20, bad TLD 0.20, ...), worst URL wins |
| x7 Logo | `signals/x7_attachment_url.py` | `LOGO_MODEL_PATH`, `LOGO_LABEL_MAP_PATH`, `LOGO_ARCH_FACTORY` | torch `state_dict` for a classifier over 64x64 RGB crops, plus a JSON label map `{"0": "none", "1": "hdfc", ...}` | each inline image decoded to RGB and resized to 64x64 | Setting-only heuristic: brand named in the text + logo shaped inline image + sending domain not entitled to that brand = 0.55, capped low because a scorer that has not looked at the image cannot honestly claim more |
| Fusion | `fusion/ebm_fusion.py` | `MODEL_PATH` | joblib dumped `interpret.glassbox.ExplainableBoostingClassifier`, interactions enabled, **must** support `explain_local()` | 14 columns in `FEATURE_NAMES` order: 7 scores then 7 `is_present` flags | Weighted logistic combination: `SIGNAL_WEIGHTS` (x2 3.6, x5 3.2, x7 3.4, x4 1.8, x3 1.6, x6 1.4, x1 0.9), `INTERACTION_WEIGHTS` (x2*x5 2.2, x2*x7 0.9, x4*x5 0.8, x6*x7 0.6), `MISSING_WEIGHTS` per signal, intercept -4.2 |
| Calibration | `fusion/ebm_fusion.py` | `CALIBRATOR_PATH` | joblib dumped isotonic regression or Platt scaler with `predict()` or `predict_proba()`, fitted on held out mail | the raw fused score | identity function (no calibration, and the report says so) |

Notes for whoever trains these:

- **x1 carries almost nothing on its own, and that is deliberate.** An attacker
  who registers their own lookalike domain publishes correct SPF and DKIM and
  authenticates perfectly. SPF and DKIM prove custody of a domain, not honesty
  of a sender. Do not "fix" x1's low weight.
- Train each signal's model on its **own** question ("does this identity hold
  up?"), not on "is this mail fraudulent". A signal that tries to make the
  verdict itself stops being separable, and then the evidence table stops
  meaning anything.
- Each `_heuristic_score()` docstring states its own known weaknesses - x5 has
  no notion of negation, quoting or non-English text; x4 treats drift and
  length as independent when they plainly are not; x7 is blind to any URL trick
  nobody has written down yet. Those are the targets to beat.

---

## 6. Campaign graph

`graph/campaign_graph.py` stores artefacts, not messages: from domain,
boundary IP, ASN, Reply-To, DKIM selector (read from x1's `details`, with a
header fallback) and every URL domain.

Artefacts are weighted by **specificity**, because joining on any shared value
would collapse the corpus into one useless campaign - every sender behind one
hosting provider shares an ASN.

| Artefact | Specificity |
| --- | --- |
| `reply_to` | 1.00 |
| `url_domain` | 0.90 |
| `dkim_selector`, `boundary_ip` | 0.80 |
| `from_domain` | 0.60 |
| `asn` | 0.30 |
| `country` | 0.05 (recorded, never used to link) |

Shared weight is summed and a link forms at `LINK_THRESHOLD = 0.6`. One high
specificity artefact links; an ASN alone does not. Shared-provider values
(`gmail.com`, `bit.ly`, `sharepoint.com`, ...) are recorded but never link.

Two backends behind one interface: an in-memory index (the default, always
works, and the source of truth for resolution) and Neo4j as a write-through
mirror for interactive analyst queries. A graph database being unreachable can
never change a verdict.

```bash
python mailguard/graph/campaign_graph.py
```

Two mails from **different** domains with **different** landing pages and
**different** DKIM selectors, sharing one ASN and one Reply-To, land in the
same campaign. A third mail sharing only the ASN correctly does not.

---

## 7. Evidence ledger

`evidence/ledger.py` is an append only JSONL file where every entry carries the
SHA-256 of the entry before it.

- `seal_raw(raw_sha256, source, operator)` records the **first** entry, before
  any parsing, and raises `ValueError` if anything else already exists for that
  case. Hashing after processing proves only that our own output is unchanged,
  not that it matches what arrived; a defence lawyer will ask when the hash was
  taken, and "after we had rewritten the headers" is not an answer.
- `append(entry)` returns the new chain head.
- `verify_chain()` returns `(intact, index of the first broken link)`. It
  catches both a reordered or deleted record (`prev_hash` mismatch) and an
  edited one (recomputed hash mismatch).

In production this is a database table with append-only grants plus an external
timestamp (RFC 3161 or a transparency log), so the chain head is anchored to a
time nobody in the organisation controls. A local file proves internal
consistency; it does not prove the file was not rebuilt from scratch.

```bash
python mailguard/evidence/ledger.py
```

Builds five entries, verifies them, tampers with entry 2, and shows
`verify_chain() -> (False, 2)`.

---

## 8. The forensic report

```bash
python mailguard/evidence/pdf_report.py
```

One command, no other part of the pipeline needed: it builds a complete fake
verdict from `tests/factory.py`, seals a temporary ledger, links two messages
into a campaign, and writes **`sample_report.pdf`** in the working directory.

Sections, in order: header (case id, timestamp, raw SHA-256), verdict block
(action in large colour coded type, probability, class), message summary,
**evidence table** (one row per signal, sorted by contribution, abstentions
marked `abstain, not scored`), model arithmetic (interaction terms, intercept,
total log-odds), origin attribution, relay trace with a visible rule at the
trust boundary, campaign linkage (omitted when there is no campaign), chain of
custody. Page number and case id on every page.

Attribution discipline: no confident location is printed for tier 2 or tier 3.
At tier 3 the report states plainly that the origin was deliberately concealed
and that the concealment is itself the finding.

If `reportlab` is not installed, the same report is written as a self-contained
HTML file at the same path with a `.html` extension and that path is returned,
so report generation never simply fails.

---

## 9. Running the tests

```bash
python tests/test_fusion.py
```

Plain asserts, no pytest required (pytest will also collect them). Coverage:
seven ok signals produce a probability in range; the fourteen value vector
carries missingness; **abstain produces a different result from a 0.0 score**;
contributions sum to the logit; authenticated BEC still BLOCKs; verdict class
picks Impersonated versus Phishing correctly; the ledger detects a tampered
entry and refuses to seal out of order; all four owned signals run on the
fixture and never raise on garbage.

## 10. Configuration

| Environment variable | Default | Effect |
| --- | --- | --- |
| `MAILGUARD_PROTECTED_DOMAINS` | built-in list | Comma separated brands x2 defends |
| `MAILGUARD_PROTECTED_DOMAINS_FILE` | unset | Same, one domain per line |
| `MAILGUARD_HISTORY_PATH` | unset | JSON sender history store read by x2 and x4 |
| `MAILGUARD_MIN_BASELINE` | `5` | Prior messages x4 needs before it stops abstaining |
| `MAILGUARD_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for x4 drift |
| `MAILGUARD_RESOLVE_REDIRECTS` | unset | Set to `1` to let x7 follow shortened links |
| `MAILGUARD_GRAPH_PATH` | unset | JSON persistence for the campaign graph |
| `MAILGUARD_NEO4J_URI` / `_USER` / `_PASSWORD` | unset | Neo4j mirror for the campaign graph |
| `MAILGUARD_LEDGER_PATH` | `mailguard_ledger.jsonl` | Default evidence ledger file |
