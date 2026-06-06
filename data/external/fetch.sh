#!/usr/bin/env bash
# Reproduce every external dataset download under data/external/.
#
# Free / public sources only. Run from anywhere; paths are resolved relative to
# this script. Idempotent: re-running re-fetches and overwrites.
#
#   bash data/external/fetch.sh
#
# Requirements: bash, curl, python3 (for ECB/CISS trim, FRED slice de-dup),
# and optionally `pdftotext` (poppler) to re-extract the Moody's table.
#
# NETWORK NOTES
#  - FRED keyless graph endpoint (fredgraph.csv) intermittently returns HTTP 504
#    on large/full-history single requests. We fetch the heavy daily series
#    (BAA10Y, VIXCLS) in multi-year SLICES and concatenate; small/monthly/weekly
#    series are fetched whole. Each fetch retries with backoff and rejects 504/HTML.
#  - FRED full HISTORY uses cosd=1990-01-01 (some series naturally start later).
#  - Some series need an authenticated FRED API key (free):
#      https://fredaccount.stlouisfed.org/apikey   -> export FRED_API_KEY=...
#    Key-only examples are printed at the end (BAMLH0A0HYM2 full history, OFRFSI).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p fdic fred ecb ratings vlab ffiec bis

# ---------- helpers ----------
# fetch_whole <url> <outfile>  (retry, reject 504/HTML/empty)
fetch_whole() {
  local url="$1" out="$2"
  for a in 1 2 3 4 5; do
    curl -sSL --max-time 60 "$url" -o "$out.tmp" 2>/dev/null
    if [ -s "$out.tmp" ] && ! grep -qiE "504|gateway time-out|<html" "$out.tmp"; then
      mv "$out.tmp" "$out"; echo "  OK  $out"; return 0
    fi
    echo "  .. retry $a/5 for $out"; rm -f "$out.tmp"; sleep 4
  done
  echo "  FAIL $out (see network notes; may need FRED API key)"; return 1
}

# fred_whole <SERIES_ID> [cosd]
fred_whole() {
  local id="$1" cosd="${2:-1990-01-01}"
  fetch_whole "https://fred.stlouisfed.org/graph/fredgraph.csv?id=${id}&cosd=${cosd}" "fred/${id}.csv"
}

# fred_sliced <SERIES_ID> <s1> <e1> <s2> <e2> ...   (concatenate windows, de-dup by date)
fred_sliced() {
  local id="$1"; shift
  local asm="fred/${id}.asm"; : > "$asm"; local hw=0 ok=1 i=0
  local pairs=("$@")
  while [ $i -lt ${#pairs[@]} ]; do
    local s="${pairs[$i]}" e="${pairs[$((i+1))]}"; i=$((i+2)); local got=0
    for a in 1 2 3 4 5; do
      curl -sSL --max-time 50 "https://fred.stlouisfed.org/graph/fredgraph.csv?id=${id}&cosd=${s}&coed=${e}" -o fred/sl.tmp 2>/dev/null
      if [ -s fred/sl.tmp ] && head -1 fred/sl.tmp | grep -q observation_date && ! grep -qiE "504|<html" fred/sl.tmp; then
        if [ $hw -eq 0 ]; then cat fred/sl.tmp >> "$asm"; hw=1; else tail -n +2 fred/sl.tmp >> "$asm"; fi
        echo "  OK  ${id} [$s..$e]"; got=1; break
      fi
      echo "  .. retry $a/5 ${id} [$s..$e]"; sleep 3
    done
    [ $got -eq 0 ] && { echo "  FAIL ${id} [$s..$e]"; ok=0; }
  done
  rm -f fred/sl.tmp
  if [ $ok -eq 1 ] && [ -s "$asm" ]; then
    python3 - "$asm" "fred/${id}.csv" <<'PY'
import sys
src,dst=sys.argv[1:3]; seen=set(); out=[]
with open(src) as f:
    out.append(f.readline().rstrip("\n"))
    for ln in f:
        d=ln.split(",",1)[0]
        if d and d!="observation_date" and d not in seen:
            seen.add(d); out.append(ln.rstrip("\n"))
open(dst,"w").write("\n".join(out)+"\n")
print(f"  -> fred/{dst.split('/')[-1]}: {len(out)-1} rows")
PY
    rm -f "$asm"
  else
    echo "  FAIL ${id} (incomplete slices)"; rm -f "$asm"
  fi
}

echo "== FDIC failed-bank events (public domain) =="
# Primary long-history file (1980-2024): S&L + GFC clusters.
curl -sSL --max-time 90 \
  "https://api.fdic.gov/banks/failures?fields=NAME,CERT,CITYST,FAILDATE,QBFDEP,QBFASSET,COST,RESTYPE,FAILYR&filters=FAILYR:%5B1980%20TO%202024%5D&limit=10000&format=csv&download=true&filename=bank-failures-1980" \
  -o fdic/failures_1980_2024.csv && echo "  OK  fdic/failures_1980_2024.csv ($(( $(wc -l < fdic/failures_1980_2024.csv) - 1 )) rows)"
# Original 2000-2024 file (kept for continuity; comment out if you only want the long file).
curl -sSL --max-time 90 \
  "https://api.fdic.gov/banks/failures?fields=NAME,CERT,CITYST,FAILDATE,QBFDEP,QBFASSET,COST,RESTYPE&filters=FAILYR:%5B2000%20TO%202024%5D&limit=10000&format=csv&download=true&filename=bank-failures" \
  -o fdic/failures.csv && echo "  OK  fdic/failures.csv ($(( $(wc -l < fdic/failures.csv) - 1 )) rows)"

echo "== FRED credit / stress series (full history; whole-series) =="
fred_whole AAA       1990-01-01   # Moody's Aaa yield (monthly, since 1919)
fred_whole BAA       1990-01-01   # Moody's Baa yield (monthly, since 1919); BAA-AAA = credit-spread stress
fred_whole KCFSI     1990-01-01   # Kansas City Fed Financial Stress Index (monthly)
fred_whole STLFSI4   1990-01-01   # St. Louis Fed Financial Stress Index v4 (weekly, since 1993)
fred_whole NFCI      1990-01-01   # Chicago Fed National Financial Conditions Index (weekly)
fred_whole ANFCI     1990-01-01   # Chicago Fed Adjusted NFCI (weekly)
# ICE BofA US HY OAS: keyless endpoint only serves 2023-06-06+ (licensed series).
# Full 1996+ history needs the FRED API key (see bottom). We still grab the keyless window:
fred_whole BAMLH0A0HYM2 1990-01-01

echo "== FRED heavy DAILY series (windowed to dodge 504; 2007+ covers 2008 & COVID) =="
# Small windows succeed even when the full-history request 504s. Add an earlier
# window (e.g. 1986-01-01 2006-12-31) if you need pre-2007 history.
fred_sliced BAA10Y 2007-01-01 2010-12-31 2011-01-01 2014-12-31 2015-01-01 2018-12-31 2019-01-01 2021-06-30 2021-07-01 2026-12-31
fred_sliced VIXCLS 2007-01-01 2010-12-31 2011-01-01 2014-12-31 2015-01-01 2018-12-31 2019-01-01 2021-06-30 2021-07-01 2026-12-31

echo "== ECB CISS (euro-area systemic stress, daily since 1980) =="
curl -sSL --max-time 90 -H "Accept: text/csv" \
  "https://data-api.ecb.europa.eu/service/data/CISS/D.U2.Z0Z.4F.EC.SS_CIN.IDX?format=csvdata" \
  -o ecb/ciss_full.csv && python3 - <<'PY'
import csv
with open("ecb/ciss_full.csv") as f, open("ecb/ciss_euro_area.csv","w",newline="") as g:
    r=csv.DictReader(f); w=csv.writer(g); w.writerow(["observation_date","CISS"]); n=0
    for row in r: w.writerow([row["TIME_PERIOD"], row["OBS_VALUE"]]); n+=1
print(f"  OK  ecb/ciss_euro_area.csv ({n} rows)")
PY
rm -f ecb/ciss_full.csv

echo "== Moody's PD-by-rating PDF + extracted table =="
curl -sSL --max-time 90 \
  "https://www.bu.edu/econ/files/2015/01/Moodys_Default_1920-2004.pdf" \
  -o ratings/Moodys_Default_1920-2004.pdf && echo "  OK  ratings/Moodys_Default_1920-2004.pdf"
# The CSV ratings/moodys_pd_by_rating.csv is hand-transcribed (year-1 column of
# Exhibits 17 & 19) and committed to the repo. To re-derive from the PDF:
#   pdftotext -layout ratings/Moodys_Default_1920-2004.pdf - | sed -n '/Exhibit 17/,/Exhibit 20/p'
# then take the "1"-year column. (Kept manual to avoid brittle PDF-layout parsing.)
if [ ! -s ratings/moodys_pd_by_rating.csv ]; then
  echo "  NOTE ratings/moodys_pd_by_rating.csv missing — it is committed in-repo; restore from git."
fi

echo
echo "== NOT auto-fetchable here (documented in CATALOG.md) =="
cat <<'NOTE'
  - NYU V-Lab SRISK : client-rendered Next.js; rows via runtime XHR. Manual export or
                      headless-browser scrape. https://vlab.stern.nyu.edu/srisk
  - FFIEC FR Y-15 / FR Y-9C : HTTP 403 to non-browser clients (bot challenge).
                      Download caret-delimited TXT via browser:
                      https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload
  - BIS consolidated banking statistics : bulk flat CSV (~85 MB) verified reachable:
                      https://data.bis.org/static/bulk/WS_CBS_PUB_csv_flat.zip
                      EU sovereign cross-holdings = BIS Quarterly Review Table 9B.
  - OFR FSI (OFRFSI) : not on FRED keyless (404). Use OFR "Download all data" CSV:
                      https://www.financialresearch.gov/financial-stress-index/
NOTE

echo
echo "== Optional: series that need a free FRED API key =="
cat <<'NOTE'
  export FRED_API_KEY=...        # https://fredaccount.stlouisfed.org/apikey
  # ICE BofA US HY OAS, FULL 1996+ history (keyless gives only 2023+):
  curl -sSL "https://api.stlouisfed.org/fred/series/observations?series_id=BAMLH0A0HYM2&api_key=${FRED_API_KEY}&file_type=csv" -o fred/BAMLH0A0HYM2_full.csv
  # OFR Financial Stress Index via FRED API (verify the id still resolves):
  curl -sSL "https://api.stlouisfed.org/fred/series/observations?series_id=OFRFSI&api_key=${FRED_API_KEY}&file_type=csv" -o fred/OFRFSI.csv
NOTE

echo
echo "Done. See data/external/CATALOG.md for provenance + model-parameter mapping."
