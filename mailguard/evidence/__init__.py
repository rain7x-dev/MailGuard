"""Evidence package: the tamper evident case file and the forensic report.

`ledger` holds the hash chained, append only record of everything done to
a message. `pdf_report` turns a verdict plus that record into the document
an investigator receives.
"""
from __future__ import annotations

from mailguard.evidence.ledger import (
    GENESIS_HASH,
    EvidenceLedger,
    append,
    case_id_for,
    compute_entry_hash,
    default_ledger,
    head,
    seal_raw,
    set_default_ledger,
    verify_chain,
)

__all__ = [
    "EvidenceLedger",
    "GENESIS_HASH",
    "append",
    "case_id_for",
    "compute_entry_hash",
    "default_ledger",
    "head",
    "seal_raw",
    "set_default_ledger",
    "verify_chain",
]
