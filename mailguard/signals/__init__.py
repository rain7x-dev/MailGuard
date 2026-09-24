"""Signals package: seven independent scorers, discovered automatically.

Every signal module in this folder follows one contract:

    SIGNAL_ID = "x1"                      # "x1" .. "x7"
    SIGNAL_NAME = "Authentication"
    def run(email: ParsedEmail) -> SignalResult: ...   # never raises

There is deliberately NO hard coded registry. Two engineers add modules to
this folder independently (x1, x3, x6 from the forensics half; x2, x4, x5,
x7 from the ML half), and a shared list would be a file both have to edit.
discover_signals() finds whatever is present instead.

A module that fails to import (typically because an optional library it
needs is not installed yet) is skipped with a logged warning and recorded
in DISCOVERY_ERRORS, so one missing dependency costs one signal, never the
whole run.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import re
from types import ModuleType

log = logging.getLogger("mailguard.signals")

_SIGNAL_MODULE = re.compile(r"^x\d+_")

# Module name -> "ExceptionType: message" for every module that failed to
# import on the last discovery, so the CLI can say why a signal is missing.
DISCOVERY_ERRORS: dict[str, str] = {}


def _sort_key(module: ModuleType) -> tuple[int, str]:
    signal_id = str(getattr(module, "SIGNAL_ID", ""))
    digits = re.sub(r"\D", "", signal_id)
    return (int(digits) if digits else 10_000, signal_id)


def discover_signals() -> list:
    """Import every x*.py module in this package and return the ones
    exposing SIGNAL_ID, SIGNAL_NAME and run().

    Modules are matched by name (`^x\\d+_`), imported one at a time, and
    returned sorted by SIGNAL_ID (x1, x2, ... x7, then x10 and beyond).
    Import failures are logged and skipped, never raised.
    """
    DISCOVERY_ERRORS.clear()
    found: list[ModuleType] = []
    for module_info in pkgutil.iter_modules(__path__):
        name = module_info.name
        if not _SIGNAL_MODULE.match(name):
            continue
        qualified = f"{__name__}.{name}"
        try:
            module = importlib.import_module(qualified)
        except Exception as exc:  # ImportError, or anything a module does at import
            DISCOVERY_ERRORS[qualified] = f"{type(exc).__name__}: {exc}"
            log.warning("signal module %s skipped: %s: %s", qualified, type(exc).__name__, exc)
            continue
        if not (
            isinstance(getattr(module, "SIGNAL_ID", None), str)
            and isinstance(getattr(module, "SIGNAL_NAME", None), str)
            and callable(getattr(module, "run", None))
        ):
            DISCOVERY_ERRORS[qualified] = "missing SIGNAL_ID, SIGNAL_NAME or run()"
            log.warning("signal module %s skipped: does not follow the signal contract", qualified)
            continue
        found.append(module)
    return sorted(found, key=_sort_key)
