"""Leave-one-domain-out rotation config for the GDRBench DG protocol."""
from __future__ import annotations

DG_DOMAINS = ["APTOS", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR"]


def dg_sources_for_target(target: str, domains=DG_DOMAINS) -> list[str]:
    if target not in domains:
        raise ValueError(f"Unknown DG target '{target}'. Expected one of {domains}.")
    return [d for d in domains if d != target]
