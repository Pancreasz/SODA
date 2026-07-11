import pytest
from soda.dg_config import DG_DOMAINS, dg_sources_for_target


def test_domains_are_the_six_dg_datasets():
    assert DG_DOMAINS == ["APTOS", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR"]


def test_sources_exclude_target_in_order():
    assert dg_sources_for_target("APTOS") == ["DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR"]
    assert dg_sources_for_target("IDRID") == ["APTOS", "DEEPDR", "FGADR", "MESSIDOR", "RLDR"]


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        dg_sources_for_target("EYEPACS")
