# tools/check_links.py
"""
Link checker + safe auto-fixer for Krop CSVs.

What it does
------------
- Scans every CSV in ./data for a 'source' column (supports multiple URLs per cell).
- Robustly checks each URL (HEAD with fallback to GET; follows redirects).
- Treats downloadable PDFs as OK when GET succeeds (even if HEAD 403/405).
- Suggests/optionally applies safe replacements for known moved pages (CFIA, EU).
- Emits a CSV report: output/link_report.csv
- With --apply, writes updated CSVs in-place (after a timestamped backup/ folder).

Usage
-----
# dry run (just create report)
python tools/check_links.py

# apply safe fixes back into the CSVs
python tools/check_links.py --apply

Dependencies
------------
pip install -r requirements.txt
"""

from __future__ import annotations
import csv
import os
import re
import sys
import time
import shutil
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTDIR = ROOT / "output"
OUTCSV = OUTDIR / "link_report.csv"

FILES = [
    "countries.csv",
    "crops.csv",
    "reg_policies.csv",
    "studies.csv",
    "incidents.csv",
    "companies.csv",
    "bans.csv",
]

URL_SPLIT_RE = re.compile(r"[;\n]+")  # split multiple links in one cell

# ---------- HTTP session (robust) ----------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        # polite UA; some gov sites block unknown/default clients
        "User-Agent": (
            "Krop-LinkChecker/1.0 (+https://example.org) "
            "Requests/%s" % requests.__version__
        )
    })
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["HEAD", "GET"])
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

session = make_session()

# ---------- helpers ----------
def normalize_url(u: str) -> str:
    return u.strip().strip("[]()<>,'\"")

def split_urls(cell: str) -> List[str]:
    if not isinstance(cell, str):
        return []
    parts = [normalize_url(p) for p in URL_SPLIT_RE.split(cell) if p.strip()]
    # drop obvious non-urls fragments
    return [p for p in parts if re.match(r"^https?://", p)]

def is_pdf_response(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    disp = (resp.headers.get("Content-Disposition") or "").lower()
    return ("application/pdf" in ctype) or ("filename=" in disp and disp.endswith(".pdf"))

@dataclass
class CheckResult:
    file: str
    row_idx: int
    context: str  # free text context (e.g., crop/decision/year), we’ll try to fill from row
    original_url: str
    final_url: str
    status_code: Optional[int]
    content: str  # OK, OK_PDF, REDIRECTED, BLOCKED, NOT_FOUND, ERROR
    note: str = ""
    suggestion: str = ""  # auto-fix suggestion (new URL or how to fix)

# ---------- known fixers (moved pages) ----------
def cfia_replacement(url: str) -> Optional[Tuple[str, str]]:
    """
    CFIA's older 'approved-events' paths 404 now.
    Use current PNT landing / approved pages instead.
    """
    if "inspection.canada.ca" in url and "approved-events" in url:
        # two stable destinations that map the old content
        # (1) PNT hub (about + navigation)
        hub = "https://inspection.canada.ca/en/plant-health/plant-varieties/novel-traits"
        # (2) page listing approvals & under review
        listing = "https://inspection.canada.ca/en/plant-health/plant-varieties/novel-traits/approved-under-review"
        # prefer the approvals listing for “policy” rows
        return listing, "CFIA 'approved-events' moved; using current approvals page."
    return None

def eu_replacement(url: str, context_text: str = "") -> Optional[Tuple[str, str]]:
    """
    Map older EU GMO authorisation links to current pages.
    """
    if "food.ec.europa.eu" in url and "genetically-modified-organisms" in url:
        # Use specific page if we can guess from context, else the general Food/Feed page.
        base_foodfeed = "https://food.ec.europa.eu/plants/genetically-modified-organisms/gmo-authorisation/authorisations-food-and-feed_en"
        base_cultivation = "https://food.ec.europa.eu/plants/genetically-modified-organisms/gmo-authorisation/cultivation_en"
        ctx = context_text.lower()
        if "cultivation" in ctx or "grown" in ctx or "field" in ctx:
            return base_cultivation, "EU cultivation page (old path moved)."
        return base_foodfeed, "EU food/feed authorisations page (old path moved)."
    # very old ec.europa.eu paths
    if "ec.europa.eu/food" in url and "genetically-modified-organisms" in url:
        base_foodfeed = "https://food.ec.europa.eu/plants/genetically-modified-organisms/gmo-authorisation/authorisations-food-and-feed_en"
        base_cultivation = "https://food.ec.europa.eu/plants/genetically-modified-organisms/gmo-authorisation/cultivation_en"
        return (base_cultivation if "authorised-gm-maize" in url else base_foodfeed,
                "EU pages reorganised; using current destination.")
    return None

def guess_replacement(url: str, context_text: str = "") -> Optional[Tuple[str, str]]:
    return cfia_replacement(url) or eu_replacement(url, context_text)

# ---------- main HTTP check ----------
def fetch_status(u: str) -> Tuple[Optional[int], str, str, str]:
    """
    Try HEAD, then GET if needed. Follow redirects.
    Returns (status_code, final_url, content_flag, note)
      content_flag in {"OK", "OK_PDF", "REDIRECTED", "BLOCKED", "NOT_FOUND", "ERROR"}
    """
    try:
        # HEAD first
        r = session.head(u, allow_redirects=True, timeout=20)
        code = r.status_code
        final = r.url

        if code in (200, 301, 302, 303, 307, 308):
            # Many servers don’t implement HEAD fully for PDFs → if we suspect a file,
            # do a lightweight GET to confirm content.
            if code == 200 and "application/pdf" in (r.headers.get("Content-Type", "").lower()):
                return code, final, "OK_PDF", "HEAD says PDF"
            if code != 200:
                return code, final, "REDIRECTED", f"Redirect chain ended with {code}"

        # If HEAD looked bad/forbidden/not allowed, try GET
        if code in (401, 403, 404, 405) or code >= 500:
            rg = session.get(u, allow_redirects=True, timeout=30, stream=True)
            code_g = rg.status_code
            final_g = rg.url
            if code_g == 200:
                # OK. If it’s a PDF (or any attachment), mark as OK_PDF.
                if is_pdf_response(rg):
                    return code_g, final_g, "OK_PDF", "GET ok; downloadable"
                return code_g, final_g, "OK", "GET ok"
            if code_g in (301, 302, 303, 307, 308):
                return code_g, final_g, "REDIRECTED", f"GET redirect ended with {code_g}"
            if code_g == 404:
                return code_g, final_g, "NOT_FOUND", "GET 404"
            if code_g in (401, 403):
                return code_g, final_g, "BLOCKED", "GET blocked (auth/forbidden)"
            return code_g, final_g, "ERROR", f"GET error {code_g}"

        # non-200 HEAD that wasn’t caught above
        if code == 404:
            return code, final, "NOT_FOUND", "HEAD 404"
        if code in (401, 403):
            return code, final, "BLOCKED", "HEAD blocked (auth/forbidden)"

        # default OK for HEAD 200
        return code, final, "OK", "HEAD ok"

    except requests.RequestException as e:
        return None, u, "ERROR", f"Exception: {type(e).__name__}"

def context_from_row(row: pd.Series) -> str:
    # Try to build a short snippet to help auto-routing EU pages, etc.
    bits = []
    for key in ("country", "crop", "policy_type", "decision", "type", "notes"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            bits.append(f"{key}={row[key]}")
    return "; ".join(bits)[:180]

def scan_file(name: str) -> List[CheckResult]:
    path = DATA / name
    results: List[CheckResult] = []
    if not path.exists():
        return results

    df = pd.read_csv(path)
    if "source" not in df.columns:
        return results

    for i, row in df.iterrows():
        urls = split_urls(str(row["source"]))
        if not urls:
            continue
        ctx = context_from_row(row)
        for u in urls:
            status, final, flag, note = fetch_status(u)
            suggestion = ""
            # Suggest replacement if 404/ERROR/BLOCKED or domain known to be moved
            if flag in ("NOT_FOUND", "ERROR", "BLOCKED") or "inspection.canada.ca" in u or "food.ec.europa.eu" in u or "ec.europa.eu/food" in u:
                repl = guess_replacement(u, ctx)
                if repl:
                    final = repl[0]
                    suggestion = repl[1]
            results.append(CheckResult(
                file=name, row_idx=i+2, context=ctx,
                original_url=u, final_url=final, status_code=status,
                content=flag, note=note, suggestion=suggestion
            ))
    return results

def backup_data_folder() -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    bdir = ROOT / "backups" / f"data-backup-{stamp}"
    bdir.parent.mkdir(exist_ok=True, parents=True)
    shutil.copytree(DATA, bdir)
    return bdir

def apply_fixes(report_df: pd.DataFrame) -> None:
    """
    Apply safe replacements back to CSVs.
    'Safe' means: suggestion present AND final_url differs AND final_url starts with:
       - https://inspection.canada.ca/
       - https://food.ec.europa.eu/
    We also de-duplicate and preserve multiple links in a cell.
    """
    ALLOWED_PREFIXES = (
        "https://inspection.canada.ca/",
        "https://food.ec.europa.eu/",
    )

    by_file: Dict[str, Dict[int, Dict[str, str]]] = {}
    for _, r in report_df.iterrows():
        if not isinstance(r.get("suggestion"), str) or not r["suggestion"]:
            continue
        new_u = str(r["final_url"] or "").strip()
        old_u = str(r["original_url"] or "").strip()
        if not new_u or new_u == old_u:
            continue
        if not any(new_u.startswith(p) for p in ALLOWED_PREFIXES):
            continue
        fname = r["file"]
        rowi = int(r["row_idx"]) - 2  # back to 0-based index
        by_file.setdefault(fname, {}).setdefault(rowi, {})[old_u] = new_u

    if not by_file:
        print("No safe fixes to apply.")
        return

    print(f"Applying fixes to {len(by_file)} file(s)...")
    for fname, rowmap in by_file.items():
        p = DATA / fname
        df = pd.read_csv(p)
        for rowi, replmap in rowmap.items():
            src = str(df.at[rowi, "source"])
            parts = split_urls(src)
            if not parts:
                continue
            # map old->new if match, then rebuild the cell preserving any extra text/format
            replaced = []
            for link in parts:
                replaced.append(replmap.get(link, link))
            # de-duplicate while preserving order
            seen = set()
            deduped = []
            for link in replaced:
                if link not in seen:
                    seen.add(link)
                    deduped.append(link)
            df.at[rowi, "source"] = "; ".join(deduped)
        df.to_csv(p, index=False)
    print("Fixes applied.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Apply safe fixes back into CSVs (backs up ./data first).")
    args = ap.parse_args()

    OUTDIR.mkdir(exist_ok=True, parents=True)

    all_rows: List[CheckResult] = []
    for f in FILES:
        print(f"Checking {f} ...")
        all_rows.extend(scan_file(f))

    cols = [
        "file", "row_idx", "context",
        "original_url", "final_url", "status_code",
        "content", "note", "suggestion",
    ]

    with OUTCSV.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({
                "file": r.file,
                "row_idx": r.row_idx,
                "context": r.context,
                "original_url": r.original_url,
                "final_url": r.final_url,
                "status_code": r.status_code if r.status_code is not None else "",
                "content": r.content,
                "note": r.note,
                "suggestion": r.suggestion
            })

    total = len(all_rows)
    bad = sum(1 for r in all_rows if r.content in ("NOT_FOUND", "ERROR", "BLOCKED"))
    print(f"✅ Link check complete: {total} links scanned, {bad} flagged.")
    print(f"Report saved to: {OUTCSV}")

    if args.apply:
        print("Creating safety backup of ./data ...")
        bdir = backup_data_folder()
        print(f"Backup created at: {bdir}")
        report_df = pd.read_csv(OUTCSV)
        apply_fixes(report_df)

if __name__ == "__main__":
    main()