# krop/tools/dedup_all.py
# Generic de-duplicator for all Phase 1 CSVs.
# Keeps ONE row per natural key (per file), merges notes, and merges/dedups sources.
# Never invents data.

from __future__ import annotations
from pathlib import Path
import csv, re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# File -> (headers, key_fields)
SCHEMAS = {
    "countries.csv": (["country","region","gmo_stance","notes","source"], ["country"]),
    "crops.csv":     (["crop","trait","description","source"], ["crop","trait"]),
    "reg_policies.csv": (["country","crop","policy_type","decision","year","notes","source"],
                         ["country","crop","policy_type","decision","year"]),
    "studies.csv":   (["study_id","crop","focus","findings","year","source"], ["study_id"]),
    "mapping.csv":   (["country","crop","prevalence","notes","source"], ["country","crop"]),
}

URL_SPLIT_RE = re.compile(r"\s*;\s*")

def norm(s: str | None) -> str:
    return (s or "").strip()

def split_sources(s: str) -> list[str]:
    if not s: return []
    return [p.strip() for p in URL_SPLIT_RE.split(s) if p.strip()]

def join_sources(urls: list[str]) -> str:
    seen, out = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return "; ".join(out)

def dedup_file(name: str) -> tuple[int, int]:
    """Return (kept, removed)."""
    headers, key_fields = SCHEMAS[name]
    path = DATA / name
    if not path.exists():
        return (0, 0)

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_key = {}
    removed = 0

    for r in rows:
        # normalize all fields we care about
        row = {h: norm(r.get(h, "")) for h in headers}
        # build key (case-insensitive)
        key = tuple(row[k].lower() for k in key_fields)

        if key not in by_key:
            by_key[key] = row
            continue

        # duplicate -> merge
        base = by_key[key]

        # merge notes (preserve both, de-dup exact substrings)
        n1, n2 = base.get("notes",""), row.get("notes","")
        if n2 and n2 not in n1:
            base["notes"] = f"{n1}  |  {n2}" if n1 else n2

        # merge sources (semicolon-separated)
        if "source" in base:
            s1 = split_sources(base.get("source",""))
            s2 = split_sources(row.get("source",""))
            base["source"] = join_sources(s1 + s2)

        # prefer numeric 'year' if one side is empty or non-numeric (only if column exists)
        if "year" in base:
            y1, y2 = base.get("year",""), row.get("year","")
            # keep a 4-digit if available
            def pick(y_old, y_new):
                if re.fullmatch(r"\d{4}", y_old): return y_old
                if re.fullmatch(r"\d{4}", y_new): return y_new
                return y_old or y_new
            base["year"] = pick(y1, y2)

        removed += 1

    # stable sort for clean diffs
    def sort_key(r: dict):
        return tuple(r.get(h,"").lower() for h in key_fields) + (r.get("year",""),)

    uniques = list(by_key.values())
    uniques.sort(key=sort_key)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in uniques:
            w.writerow({h: r.get(h,"") for h in headers})

    return (len(uniques), removed)

def main():
    total_removed = 0
    for name in SCHEMAS:
        kept, removed = dedup_file(name)
        if kept or removed:
            print(f"✅ {name}: kept {kept}, removed {removed}")
            total_removed += removed
    if total_removed == 0:
        print("✨ No duplicates found across CSVs.")

if __name__ == "__main__":
    main()