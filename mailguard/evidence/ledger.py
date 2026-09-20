"""Tamper evident case file: a hash chained, append only evidence ledger.

Every action MailGuard takes on a message is written here as one line, and
every line carries the SHA-256 of the line before it. Changing any earlier
record changes its hash, which breaks the link stored in the next record,
and every record after that. So the ledger cannot prove that nobody tried
to alter it, but it can prove that nobody succeeded without leaving the
break visible, and it can say exactly where the break is.

ORDER MATTERS, WHICH IS WHY seal_raw() EXISTS

The first entry for a message must be the hash of the bytes as they
arrived, recorded before any parsing, normalisation or decoding. Hashing
after processing proves nothing about what actually arrived: it proves
only that the pipeline's own output has not changed since the pipeline
produced it, which is a claim about this tool rather than about the
evidence. A defence lawyer will ask when the hash was taken, and "after
we had already rewritten the headers" is not an answer. seal_raw()
therefore refuses to run after other entries for the same message exist.

Storage is a JSONL file, one entry per line, opened in append mode. In
production this is a database table with append-only grants plus an
external timestamp (an RFC 3161 timestamping authority, or a transparency
log) so that the chain head is anchored to a time nobody in this
organisation controls. A local file proves internal consistency; it does
not prove the whole file was not rebuilt from scratch.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

if __package__ in (None, ""):  # pragma: no cover - script convenience only
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

# The genesis link: the prev_hash of the very first entry in a file.
GENESIS_HASH: str = "0" * 64

# Default ledger location, overridable per instance or by environment.
DEFAULT_LEDGER_PATH: str = os.environ.get("MAILGUARD_LEDGER_PATH", "mailguard_ledger.jsonl")

# Fields that take part in an entry's hash. entry_hash itself cannot, and
# nothing may be added to an entry after it is hashed.
HASHED_FIELDS: tuple[str, ...] = ("index", "timestamp", "type", "case_id", "prev_hash", "payload")


def case_id_for(raw_sha256: str) -> str:
    """Stable case id derived from the raw message hash.

    Derived rather than assigned so that the same bytes always produce the
    same case id, on any host, without a counter or a database.
    """
    digest = (raw_sha256 or "").strip().lower()
    return f"MG-{digest[:12].upper()}" if digest else "MG-UNKNOWN"


def _canonical(entry: dict[str, Any]) -> bytes:
    """Deterministic bytes for hashing: sorted keys, no incidental spacing."""
    subset = {field: entry.get(field) for field in HASHED_FIELDS}
    return json.dumps(subset, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def compute_entry_hash(entry: dict[str, Any]) -> str:
    """SHA-256 of an entry's hashed fields."""
    return hashlib.sha256(_canonical(entry)).hexdigest()


class EvidenceLedger:
    """An append only, hash chained ledger backed by a JSONL file."""

    def __init__(self, path: str = "") -> None:
        self.path = path or DEFAULT_LEDGER_PATH

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def entries(self) -> list[dict[str, Any]]:
        """Every entry in file order. A malformed line is kept as-is.

        Malformed lines are returned rather than skipped, because a
        corrupt line is exactly the thing verify_chain() has to be able
        to report. Quietly dropping it would hide the tampering.
        """
        if not os.path.exists(self.path):
            return []
        result: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        raise ValueError("entry is not an object")
                    result.append(parsed)
                except ValueError:
                    result.append({"_unparseable": True, "_line_number": line_number, "_raw": line})
        return result

    def head(self) -> str:
        """Hash of the most recent entry, or the genesis hash if empty."""
        entries = self.entries()
        if not entries:
            return GENESIS_HASH
        return str(entries[-1].get("entry_hash") or "")

    def count(self) -> int:
        return len(self.entries())

    def entries_for_case(self, case_id: str) -> list[dict[str, Any]]:
        return [entry for entry in self.entries() if entry.get("case_id") == case_id]

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def append(self, entry: dict[str, Any]) -> str:
        """Append an entry to the chain and return the new chain head.

        The caller supplies `type`, `case_id` and whatever payload fields
        belong to the event; index, timestamp, prev_hash and entry_hash are
        set here, because an entry that can choose its own position in the
        chain is not a chain.
        """
        existing = self.entries()
        index = len(existing)
        prev_hash = str(existing[-1].get("entry_hash")) if existing else GENESIS_HASH

        payload = dict(entry or {})
        entry_type = str(payload.pop("type", "note"))
        case_id = str(payload.pop("case_id", "") or "")
        # Let a caller pass an explicit event time; default to now.
        timestamp = str(payload.pop("timestamp", "") or datetime.now(timezone.utc).isoformat(timespec="seconds"))

        record: dict[str, Any] = {
            "index": index,
            "timestamp": timestamp,
            "type": entry_type,
            "case_id": case_id,
            "prev_hash": prev_hash,
            "payload": payload,
        }
        record["entry_hash"] = compute_entry_hash(record)

        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return str(record["entry_hash"])

    def seal_raw(self, raw_sha256: str, source: str, operator: str) -> str:
        """Record the hash of the message as it arrived, before parsing.

        This must be the FIRST entry for a message. Hashing after
        processing proves only that our own output is unchanged, not that
        it matches what arrived, so the ordering is the property that
        makes the rest of the ledger worth anything. Raises ValueError if
        anything else has already been recorded for this case.

        Returns the new chain head.
        """
        case_id = case_id_for(raw_sha256)
        prior = self.entries_for_case(case_id)
        if prior:
            raise ValueError(
                f"cannot seal {case_id}: {len(prior)} entries already exist for it. "
                "The raw seal has to be the first record for a message, otherwise "
                "it says nothing about what actually arrived."
            )
        return self.append(
            {
                "type": "raw_seal",
                "case_id": case_id,
                "raw_sha256": (raw_sha256 or "").strip().lower(),
                "source": source,
                "operator": operator,
                "note": "hash of the message bytes as received, taken before any parsing",
            }
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def verify_chain(self) -> tuple[bool, int | None]:
        """Walk the chain. Returns (intact, index of the first broken link).

        A link is broken when an entry cannot be parsed, when its
        prev_hash does not match the previous entry's hash, when its index
        is out of order, or when its stored hash does not match a
        recomputation of its own contents. The second and fourth cases are
        the two halves of the guarantee: the first catches a deleted or
        reordered record, the second catches an edited one.
        """
        entries = self.entries()
        if not entries:
            return True, None
        previous_hash = GENESIS_HASH
        for position, entry in enumerate(entries):
            if entry.get("_unparseable"):
                return False, position
            if int(entry.get("index", -1)) != position:
                return False, position
            if str(entry.get("prev_hash")) != previous_hash:
                return False, position
            if compute_entry_hash(entry) != str(entry.get("entry_hash")):
                return False, position
            previous_hash = str(entry.get("entry_hash"))
        return True, None

    def verification_report(self) -> dict[str, Any]:
        """Verification result plus the numbers a report needs to print."""
        intact, broken_index = self.verify_chain()
        entries = self.entries()
        seal = next((entry for entry in entries if entry.get("type") == "raw_seal"), None)
        return {
            "path": self.path,
            "intact": intact,
            "broken_at": broken_index,
            "entry_count": len(entries),
            "head": self.head(),
            "sealed_at": (seal or {}).get("timestamp"),
            "seal_hash": (seal or {}).get("entry_hash"),
        }


# ----------------------------------------------------------------------
# Module level convenience API over one default ledger. The class is the
# real interface; these exist so a caller with one ledger per host does
# not have to thread an object through every layer.
# ----------------------------------------------------------------------
_DEFAULT_LEDGER: Optional[EvidenceLedger] = None


def default_ledger() -> EvidenceLedger:
    """The process wide ledger at DEFAULT_LEDGER_PATH."""
    global _DEFAULT_LEDGER
    if _DEFAULT_LEDGER is None:
        _DEFAULT_LEDGER = EvidenceLedger(DEFAULT_LEDGER_PATH)
    return _DEFAULT_LEDGER


def set_default_ledger(path: str) -> EvidenceLedger:
    """Point the module level functions at a different file."""
    global _DEFAULT_LEDGER
    _DEFAULT_LEDGER = EvidenceLedger(path)
    return _DEFAULT_LEDGER


def seal_raw(raw_sha256: str, source: str, operator: str) -> str:
    """Record the FIRST entry for a message, before any parsing.

    See EvidenceLedger.seal_raw: hashing after processing proves nothing
    about what actually arrived, so this has to come first.
    """
    return default_ledger().seal_raw(raw_sha256, source, operator)


def append(entry: dict[str, Any]) -> str:
    """Append an entry to the hash chain and return the new chain head."""
    return default_ledger().append(entry)


def verify_chain() -> tuple[bool, int | None]:
    """Walk the chain, returning (intact, index of first broken link)."""
    return default_ledger().verify_chain()


def head() -> str:
    """Current chain head hash."""
    return default_ledger().head()


if __name__ == "__main__":  # pragma: no cover - demonstration
    import tempfile

    print("MailGuard evidence ledger demonstration")
    print("=" * 78)

    directory = tempfile.mkdtemp(prefix="mailguard-ledger-")
    path = os.path.join(directory, "case.jsonl")
    ledger = EvidenceLedger(path)

    raw_hash = hashlib.sha256(b"From: attacker@hdfc-verify.example\r\n\r\nPay us.").hexdigest()
    case = case_id_for(raw_hash)
    print(f"ledger file : {path}")
    print(f"case id     : {case}")
    print()

    print("building a chain of five entries")
    print("-" * 78)
    chain_head = ledger.seal_raw(raw_hash, source="smtp://mx1.victim.example", operator="analyst.rk")
    print(f"  0 raw_seal        head {chain_head[:16]}...")
    chain_head = ledger.append({"type": "parsed", "case_id": case, "mime_parts": 4, "urls_found": 2})
    print(f"  1 parsed          head {chain_head[:16]}...")
    chain_head = ledger.append(
        {"type": "signals", "case_id": case, "x2": 0.91, "x5": 0.86, "x4": "abstain"}
    )
    print(f"  2 signals         head {chain_head[:16]}...")
    chain_head = ledger.append(
        {"type": "verdict", "case_id": case, "probability": 0.98, "action": "BLOCK"}
    )
    print(f"  3 verdict         head {chain_head[:16]}...")
    chain_head = ledger.append({"type": "report", "case_id": case, "artefact": "report.pdf"})
    print(f"  4 report          head {chain_head[:16]}...")
    print()

    intact, broken_at = ledger.verify_chain()
    print(f"verify_chain() -> ({intact}, {broken_at})")
    if intact and broken_at is None:
        print("  PASS: five entries, every prev_hash matches, every hash recomputes.")
    else:
        print("  unexpected: a freshly built chain should verify.")
    print()

    print("now deliberately tampering with entry 2, changing x2 from 0.91 to 0.10")
    print("-" * 78)
    lines = open(path, "r", encoding="utf-8").read().splitlines()
    tampered = json.loads(lines[2])
    tampered["payload"]["x2"] = 0.10
    # The forger rewrites the record but cannot recompute the rest of the
    # chain, which is the entire point of chaining.
    lines[2] = json.dumps(tampered, sort_keys=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    intact, broken_at = ledger.verify_chain()
    print(f"verify_chain() -> ({intact}, {broken_at})")
    if not intact and broken_at == 2:
        print("  PASS: the ledger detected the edit and named entry 2 as the first")
        print("        broken link. Entries 3 and 4 are now unverifiable too, because")
        print("        each one's prev_hash points at the hash entry 2 used to have.")
    else:
        print(f"  unexpected result: intact={intact}, broken_at={broken_at}")
    print()

    report = ledger.verification_report()
    print("verification report:")
    for key in ("path", "intact", "broken_at", "entry_count", "head", "sealed_at"):
        print(f"  {key:<12} {report[key]}")
