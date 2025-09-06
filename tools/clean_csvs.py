# krop/tools/clean_csvs.py
# 👩‍🏫 Friendly cleaner robot for your CSV shelves.
# - Never invents data.
# - Fixes whitespace, quotes, commas.
# - Ensures 'year' is digits only (else blank).
# - Splits sources by ';', trims each.
# - Any non-URL "source" text is moved into 'notes' (to preserve info),
#   and 'source' is cleared so validator will still flag it for a real link.
from __future__ import annotations
from pathlib import Path
import csv, re, shutil
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BACKUP_DIR = ROOT / "backups"

URL_RE = re.compile(r"^https?://", re.IGNORECASE)
DIGITS_RE = re.compile(r"^\d{4}$")

FILES = {
    "countries.csv": ["country","region","gmo_stance","notes","source"],
    "crops.csv": ["crop","trait","description","source"],
    "reg_policies.csv": ["country","crop","policy_type","decision","year","notes","source"],
    "studies.csv": ["study_id","crop","focus","findings","year","source"],
    "mapping.csv": ["country","crop","prevalence","notes","source"],
}

def backup_all():
    BACKUP_DIR.mkdir(exist_ok=True, parents=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = BACKUP_DIR / f"data-backup-{stamp}"
    dest.mkdir()
    for name in FILES:
        src = DATA / name
        if src.exists():
            shutil.copy2(src, dest / name)
    return dest

def normalize_spaces(s: str|None) -> str:
    if s is None:
        return ""
    # collapse weird spaces, strip
    s = s.replace("\u00A0", " ")  # non-breaking space -> space
    s = s.strip()
    # remove accidental surrounding quotes like ""text""
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1].strip()
    return s

def clean_year(val: str) -> str:
    v = normalize_spaces(val)
    # Keep only 4-digit year; otherwise blank
    m = re.search(r"\b(\d{4})\b", v)
    return m.group(1) if m else ""

def split_sources(val: str) -> list[str]:
    # Split by ';' or ',' but prefer ';'
    raw = [p.strip() for p in re.split(r";|,(?=\s*https?://)", val or "") if p.strip()]
    return raw

def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))

def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_rows(path: Path, headers: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            # ensure only known headers
            w.writerow({h: r.get(h, "") for h in headers})

def clean_file(name: str) -> dict:
    path = DATA / name
    if not path.exists():
        return {"file": name, "status": "missing"}

    headers = FILES[name]
    raw_rows = load_rows(path)
    cleaned = []
    stats = {
        "file": name,
        "rows": len(raw_rows),
        "trimmed": 0,
        "year_fixed": 0,
        "moved_source_to_notes": 0,
        "sources_split": 0,
        "notes_appended": 0,
    }

    for r in raw_rows:
        # start from clean dict with known headers only
        row = {h: normalize_spaces(r.get(h)) for h in headers}

        # Basic trim stat
        if any((r.get(h) or "").strip() != row[h] for h in headers):
            stats["trimmed"] += 1

        # Year normalization
        if "year" in row:
            old = row["year"]
            new = clean_year(old)
            if new != old:
                row["year"] = new
                stats["year_fixed"] += 1

        # Source handling
        if "source" in row:
            sources = split_sources(row["source"])
            urls = [s for s in sources if is_url(s)]
            non_urls = [s for s in sources if s and not is_url(s)]

            # Move any non-URL source text into notes (so you can see what to replace)
            if non_urls:
                note_add = f"Source text moved here: { '; '.join(non_urls) }"
                if "notes" in row and row["notes"]:
                    row["notes"] = f"{row['notes']}  |  {note_add}"
                elif "notes" in row:
                    row["notes"] = note_add
                stats["notes_appended"] += 1
                stats["moved_source_to_notes"] += len(non_urls)

            # Write URLs back as semicolon-separated list
            if urls:
                row["source"] = "; ".join(urls)
                stats["sources_split"] += 1
            else:
                # leave empty to make validator flag it for a real link
                row["source"] = ""

        cleaned.append(row)

    write_rows(path, headers, cleaned)
    stats["status"] = "ok"
    return stats

def main():
    print(f"🧹 Cleaning CSVs in: {DATA}")
    backup_path = backup_all()
    print(f"📦 Backup created at: {backup_path}")

    results = []
    for name in FILES:
        res = clean_file(name)
        results.append(res)

    print("\n===== Summary =====")
    for r in results:
        if r.get("status") == "missing":
            print(f"⚠️  {r['file']}: missing")
        else:
            print(
                f"✅ {r['file']} — rows: {r['rows']}, "
                f"trimmed: {r['trimmed']}, year_fixed: {r['year_fixed']}, "
                f"sources_split: {r['sources_split']}, "
                f"moved_source_to_notes: {r['moved_source_to_notes']}, "
                f"notes_appended: {r['notes_appended']}"
            )
    print("\n✨ Done. Now run the validator:" 
          "\n   python tools/validate_data.py")
    

# Auto-dedupe all CSVs after cleaning
from dedup_all import main as dedup_all
dedup_all()
if __name__ == "__main__":
    main()
    