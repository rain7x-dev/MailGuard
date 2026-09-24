"""MailGuard AI: decide whether an email is fraudulent and how far its sender can honestly be traced.

The package is built in two halves that meet at `mailguard.core.models`:

  forensics half   ingest, parsing, the trust boundary walk, origin
                   attribution, signals x1 / x3 / x6, and the CLI
  ML half          signals x2 / x4 / x5 / x7, fusion, the campaign graph,
                   the evidence ledger and the PDF report

Nothing is imported here on purpose: importing the package must stay cheap
and must never fail because one half or an optional library is missing.
"""
from __future__ import annotations

__version__ = "0.1.0"
