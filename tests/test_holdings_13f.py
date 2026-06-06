from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systemic_risk.data_network.sources.holdings_13f import (
    ColumnSpec,
    cosine_overlap,
    directed_fire_sale_matrix,
    holdings_matrix,
    liquidity_weighted_overlap,
    load_holdings,
    panel_from_frame,
    rdate_to_yearqtr,
    sample_holdings_csv,
    synthetic_holdings_panel,
    validated_overlap_network,
)


# --- column resolution ------------------------------------------------------- #
def test_column_spec_detects_standard_names() -> None:
    spec = ColumnSpec.detect(["CIK", "CUSIP", "VALUE", "SSHPRNAMT", "PERIOD"])
    assert spec.institution == "CIK"
    assert spec.asset == "CUSIP"
    assert spec.value == "VALUE"
    assert spec.shares == "SSHPRNAMT"
    assert spec.quarter == "PERIOD"


def test_column_spec_detects_real_edgar_schema() -> None:
    # The real holdings.csv: asset id is CRSP permno, quarter is rdate.
    spec = ColumnSpec.detect(["cik", "rdate", "fdate", "form", "permno", "shares",
                              "value", "accession"])
    assert spec.institution == "cik"
    assert spec.asset == "permno"
    assert spec.quarter == "rdate"
    assert spec.value == "value"
    assert spec.shares == "shares"


def test_column_spec_raises_without_required_columns() -> None:
    with pytest.raises(KeyError):
        ColumnSpec.detect(["name", "value"])  # no asset / institution


@pytest.mark.parametrize(
    "rdate,expected",
    [("2008-03-31", 2008.0), ("2008-06-30", 2008.25),
     ("2008-09-30", 2008.5), ("2008-12-31", 2008.75)],
)
def test_rdate_to_yearqtr(rdate: str, expected: float) -> None:
    assert rdate_to_yearqtr(rdate) == expected


def test_sample_holdings_csv_filters_quarter_and_top_filers(tmp_path) -> None:
    # Tiny stand-in for the giant holdings.csv, in the real EDGAR-Parsing schema.
    rows = []
    for cik, permno, value, rdate in [
        ("A", "1", "900", "2008-09-30"), ("A", "2", "900", "2008-09-30"),
        ("B", "1", "50", "2008-09-30"),  ("C", "3", "10", "2008-09-30"),
        ("A", "1", "999", "2007-06-30"),  # other quarter -> excluded
    ]:
        rows.append({"cik": cik, "rdate": rdate, "fdate": rdate, "form": "13F-HR",
                     "permno": permno, "shares": "1", "value": value,
                     "accession": "x"})
    src = tmp_path / "holdings.csv"
    pd.DataFrame(rows).to_csv(src, index=False)

    dst = sample_holdings_csv(src, tmp_path / "slice.csv", rdates=("2008-09-30",),
                              top_institutions=2, chunksize=2)
    out = pd.read_csv(dst, dtype=str)
    assert set(out["rdate"].unique()) == {"2008-09-30"}        # only the wanted quarter
    assert set(out["cik"].unique()) == {"A", "B"}              # top-2 by AUM (C dropped)


def test_sample_holdings_csv_reports_available_when_quarter_missing(tmp_path) -> None:
    src = tmp_path / "holdings.csv"
    pd.DataFrame([{"cik": "A", "rdate": "2008-09-30", "permno": "1",
                   "shares": "1", "value": "10"}]).to_csv(src, index=False)
    with pytest.raises(ValueError, match="Available report dates"):
        sample_holdings_csv(src, tmp_path / "s.csv", rdates=("1999-12-31",))


def test_load_holdings_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_holdings(tmp_path / "nope.csv")


# --- a deterministic two-block panel for exact SVN behaviour ----------------- #
def _two_block_panel():
    rows = []
    block1 = list(range(1, 16))    # assets 1..15
    block2 = list(range(16, 31))   # assets 16..30
    holdings = {
        "A": block1,
        "B": block1,   # same block -> heavy overlap with A
        "C": block2,
        "D": block2,   # same block -> heavy overlap with C
    }
    for cik, assets in holdings.items():
        for a in assets:
            rows.append({"cik": cik, "cusip": f"CU{a:03d}", "value": 100.0 + a,
                         "quarter": "2008Q3"})
    return panel_from_frame(pd.DataFrame(rows))


def test_holdings_matrix_shape_and_nonneg() -> None:
    mat = holdings_matrix(_two_block_panel())
    assert mat.H.shape == (4, 30)
    assert np.all(mat.H >= 0)
    assert set(mat.institutions) == {"A", "B", "C", "D"}


def test_validated_overlap_recovers_blocks() -> None:
    mat = holdings_matrix(_two_block_panel())
    adjacency, pvalues = validated_overlap_network(mat, alpha=0.05)
    idx = {c: i for i, c in enumerate(mat.institutions)}
    # Within-block heavy overlaps are validated; cross-block (no common assets) are not.
    assert adjacency[idx["A"], idx["B"]] == 1
    assert adjacency[idx["C"], idx["D"]] == 1
    assert adjacency[idx["A"], idx["C"]] == 0
    assert adjacency[idx["B"], idx["D"]] == 0
    assert np.array_equal(adjacency, adjacency.T)
    assert np.all(np.diag(adjacency) == 0)


def test_cosine_overlap_properties() -> None:
    mat = holdings_matrix(_two_block_panel())
    overlap = cosine_overlap(mat)
    assert np.allclose(overlap, overlap.T)
    assert np.all(np.diag(overlap) == 0)
    assert overlap.min() >= 0 and overlap.max() <= 1
    idx = {c: i for i, c in enumerate(mat.institutions)}
    assert overlap[idx["A"], idx["B"]] > overlap[idx["A"], idx["C"]]


# --- fire-sale matrices ------------------------------------------------------ #
def test_liquidity_weighted_overlap_symmetric_and_illiquidity_matters() -> None:
    mat = holdings_matrix(_two_block_panel())
    base = liquidity_weighted_overlap(mat)
    assert np.allclose(base, base.T)
    assert np.all(np.diag(base) == 0)
    ill = np.linspace(1.0, 5.0, mat.n_assets)
    weighted = liquidity_weighted_overlap(mat, illiquidity=ill)
    assert not np.allclose(base, weighted)


def test_directed_fire_sale_is_asymmetric() -> None:
    # Different book sizes -> loss-to-j-when-i-sells differs by direction.
    rows = [
        {"cik": "BIG", "cusip": "CU1", "value": 1000.0, "quarter": "q"},
        {"cik": "BIG", "cusip": "CU2", "value": 1000.0, "quarter": "q"},
        {"cik": "SML", "cusip": "CU1", "value": 10.0, "quarter": "q"},
    ]
    mat = holdings_matrix(panel_from_frame(pd.DataFrame(rows)))
    F = directed_fire_sale_matrix(mat)
    assert np.all(np.diag(F) == 0)
    assert not np.allclose(F, F.T)


# --- synthetic generator smoke ---------------------------------------------- #
def test_synthetic_panel_builds_overlap_network() -> None:
    panel = synthetic_holdings_panel(n_institutions=24, n_assets=60, n_blocks=3, seed=1)
    mat = holdings_matrix(panel, min_assets_held=3, min_positions=2)
    assert mat.n_institutions >= 2 and mat.n_assets >= 1
    adjacency, _ = validated_overlap_network(mat)
    assert adjacency.shape == (mat.n_institutions, mat.n_institutions)
    assert adjacency.sum() > 0   # block structure yields some validated links


def test_synthetic_panel_is_deterministic() -> None:
    a = synthetic_holdings_panel(seed=7).df
    b = synthetic_holdings_panel(seed=7).df
    pd.testing.assert_frame_equal(a, b)
