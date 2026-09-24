"""Forensics package: the trust boundary walk and three tier origin attribution.

`trust_boundary` decides which Received hops were written by servers we
control and which are attacker writable. `attribution` takes the one
address we can stand behind and says, in one of three tiers, how far the
sender can honestly be traced.
"""
from __future__ import annotations
