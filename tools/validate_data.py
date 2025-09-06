# tools/validate_data.py
# Validates all Phase 1 CSVs (countries, crops, reg_policies, studies, mapping, bans).
# Checks: required columns, empty source, URL format, numeric year (where present),
# and duplicates using a natural key per file.

from __future__ import annotations
from pathlib import Path
import csv
import re
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

URL_RE = re.compile(r"^https?://", re.I)
YEAR_RE = re.compile(r"^\d{4}$")

# file -> (required_headers, key_fields, check_year_bool)
FILES = {
    "countries.csv": (["country","region","gmo_stance","notes","source"], ["country"], False),
    "crops.csv":     (["crop","trait","description","source"], ["crop","trait"], False),
    "reg_policies.csv": (
        ["country","crop","policy_type","decision","year","notes","source"],
        ["country","crop","policy_type","decision","year"],
        True
    ),
    "studies.csv":   (["study_id","crop","focus","findings","year","source"], ["study_id"], True),
    "mapping.csv":   (["country","crop","prevalence","notes","source"], ["country","crop"], False),
    # ✅ new
    "bans.csv":      (["country","scope","type","year","notes","source"], ["country","scope","type","year"], True),

    "incidents.csv": (
        ["country","year","type","description","source"],
        ["country","year","type"],
        True
    ),
    "companies.csv": (
        ["company","product","crop","trait","year","notes","source"],
        ["company","product","crop"],
        True
    ),    
    
}

def looks_like_url(s: str) -> bool:
    return bool(URL_RE.match(s or ""))

def split_urls(s: str) -> list[str]:
    # sources are semicolon-separated
    return [p.strip() for p in (s or "").split(";") if p.strip()]

def read_rows(name: str) -> list[dict]:
    path = DATA / name
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def validate_file(name: str) -> int:
    req, key_fields, check_year = FILES[name]
    rows = read_rows(name)
    bad = 0

    # 1) header check
    headers = rows[0].keys() if rows else req
    missing = [h for h in req if h not in headers]
    if missing:
        print(f"❌ {name}: missing columns {missing}")
        bad += 1

    # 2) per-row checks
    for i, r in enumerate(rows, start=2):  # +1 for header, +1 to be human-friendly
        src = (r.get("source") or "").strip()
        if not src:
            print(f"❌ {name} row {i}: empty source")
            bad += 1
        else:
            for u in split_urls(src):
                if not looks_like_url(u):
                    print(f"❌ {name} row {i}: not a URL -> {u}")
                    bad += 1

        if check_year:
            y = (r.get("year") or "").strip()
            if not YEAR_RE.match(y):
                print(f"❌ {name} row {i}: year not numeric -> '{y}'")
                bad += 1

    # 3) duplicate check
    seen = defaultdict(list)
    for idx, r in enumerate(rows, start=2):
        k = tuple((r.get(f) or "").strip().lower() for f in key_fields)
        seen[k].append(idx)
    for k, idxs in seen.items():
        if len(idxs) > 1:
            print(f"❌ {name}: duplicate entries at rows {idxs} -> {k}")
            bad += 1

    if bad == 0:
        print(f"✅ Checked {name}: {len(rows)} row(s)")
    return bad

def main():
    print("🔎 Validating CSVs…\n")
    total_bad = 0
    for name in FILES:
        try:
            total_bad += validate_file(name)
        except FileNotFoundError:
            print(f"ℹ️  Skipping {name} (not found)")
    if total_bad:
        print(f"\n❌ Validation finished with {total_bad} issue(s). Fix before committing.")
    else:
        print("\n🟢 All checks passed.")

if __name__ == "__main__":
    main()
        # --- Bans summary by region ---
    if (DATA / "bans.csv").exists() and (DATA / "countries.csv").exists():
        rows = read_rows("bans.csv")
        countries = {r["country"].strip().lower(): r["region"].strip() for r in read_rows("countries.csv")}
        counts = defaultdict(int)
        missing = []
        for r in rows:
            c = (r["country"] or "").strip().lower()
            region = countries.get(c)
            if region:
                counts[region] += 1
            else:
                missing.append(r["country"])
        print("\n🌍 Ban summary by region:")
        for reg, n in counts.items():
            print(f"  {reg}: {n} ban(s)")
        if missing:
            print(f"⚠️ Countries missing region info in countries.csv: {missing}")

