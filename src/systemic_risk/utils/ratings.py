"""Shared one-year default probabilities by whole-letter credit rating.

Single source for the rating -> PD map used by both the synthetic generator
(``data/synthetic.py``) and the empirical pipeline (``data_network/estimate.py``).
Prefers the committed Moody's table; falls back to literature defaults.
"""

from __future__ import annotations

from pathlib import Path

# Literature-default 1-year PDs (S&P/Moody's annual default studies; mid-band values).
# Used when the Moody's CSV is absent.
RATING_PD_DEFAULT: dict[str, float] = {
    "AAA": 0.0001,
    "AA": 0.0004,
    "A": 0.0008,
    "BBB": 0.0025,
    "BB": 0.0140,
    "B": 0.0550,
    "CCC": 0.2200,
}

# Moody's whole-letter (Exhibit 17) keys -> our S&P-style whole-letter buckets.
_MOODYS_TO_BUCKET: dict[str, str] = {
    "Aaa": "AAA", "Aa": "AA", "A": "A", "Baa": "BBB",
    "Ba": "BB", "B": "B", "Caa-C": "CCC",
}

# Committed Moody's PD table (Corporate Default & Recovery Rates 1920-2004, Exhibit 17),
# four levels up from this file.
RATING_PD_CSV = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "external"
    / "ratings"
    / "moodys_pd_by_rating.csv"
)


def load_rating_pd(path: str | Path | None = None) -> tuple[dict[str, float], str]:
    """Return ``(bucket -> 1-year PD, source)`` preferring the real Moody's table.

    Reads only the whole-letter (Exhibit 17) rows so every PD is on one consistent
    scale; near-zero high-grade PDs are floored at 1e-5 so logit fields stay finite
    downstream. Any parse problem falls back to the literature defaults.
    """
    csv_path = Path(path) if path is not None else RATING_PD_CSV
    table = dict(RATING_PD_DEFAULT)
    source = "literature defaults (S&P/Moody's annual default studies)"
    if not csv_path.exists():
        return table, source
    try:
        rows: dict[str, float] = {}
        for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
            if not line.strip():
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            rating, pd_str, src = parts[0].strip(), parts[1], parts[2]
            if "Exhibit 17" not in src:
                continue
            bucket = _MOODYS_TO_BUCKET.get(rating)
            if bucket and bucket not in rows:
                rows[bucket] = float(pd_str)
        if rows:
            table.update({k: max(v, 1e-5) for k, v in rows.items()})
            source = (
                "Moody's Corporate Default & Recovery Rates 1920-2004, "
                "Exhibit 17 (whole-letter, Year-1)"
            )
    except (OSError, ValueError):
        return dict(RATING_PD_DEFAULT), source
    return table, source
