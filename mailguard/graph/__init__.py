"""Graph package: link separate attacks that share hidden infrastructure.

`campaign_graph` holds the artefact graph, the specificity weighting that
decides what counts as a link, and both backends (in-memory by default,
Neo4j as a write-through mirror when configured).
"""
from __future__ import annotations

from mailguard.graph.campaign_graph import (
    LINK_SPECIFICITY,
    LINK_THRESHOLD,
    CampaignGraph,
    extract_artefacts,
    linkable_artefacts,
)

__all__ = [
    "CampaignGraph",
    "LINK_SPECIFICITY",
    "LINK_THRESHOLD",
    "extract_artefacts",
    "linkable_artefacts",
]
