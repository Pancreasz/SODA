from pathlib import Path

from soda.data.verify_layout import verify_split


def _make_tree(tmp_path: Path):
    root = tmp_path / "data"
    (root / "IDRID" / "nodr").mkdir(parents=True)
    img = root / "IDRID" / "nodr" / "a.jpg"
    img.write_bytes(b"x")
    split = tmp_path / "IDRID_train.txt"
    split.write_text("IDRID/nodr/a.jpg 0\nIDRID/nodr/missing.jpg 2\n")
    return split, root


def test_reports_total_and_missing(tmp_path):
    split, root = _make_tree(tmp_path)
    rep = verify_split(split, root)
    assert rep.total == 2
    assert rep.missing == ["IDRID/nodr/missing.jpg"]
    assert rep.label_counts == {0: 1, 2: 1}
