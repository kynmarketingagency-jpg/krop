# krop/tools/make_country_research_file.py
# 👩‍🏫 This helper makes a research checklist markdown file for a country.
# It does NOT touch your data CSVs (so no fake rows).
from pathlib import Path
import sys
from datetime import date

TEMPLATE = """# {country} — GMO Research Checklist
_Last updated: {today}_

## 1) Quick facts (fill with sources)
- National biosafety authority:
- Overall GMO stance (cultivation, food/feed imports):
- Labeling rules:

## 2) Approved or banned GM crops (each bullet must have a source link)
- Crop: 
  - Trait(s): 
  - Decision (Approved/Banned/Restricted): 
  - Year: 
  - Notes:
  - **Sources**: 

## 3) Key policies / regulations
- Law/Regulation:
- Agency:
- **Sources**:

## 4) Studies (health/environment/agronomy)
- Study ID:
- Focus:
- Findings (neutral wording):
- Year:
- **Source**:

## 5) Mapping (prevalence/adoption)
- Prevalence:
- Notes:
- **Source**:

## 6) To-verify list
- [ ] Item + link

## 7) Final data pasted into CSVs?
- [ ] countries.csv
- [ ] crops.csv
- [ ] reg_policies.csv
- [ ] studies.csv
- [ ] mapping.csv
"""

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m krop.tools.make_country_research_file \"Country Name\"")
        sys.exit(1)

    country = sys.argv[1].strip()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{country}.md"

    if path.exists():
        print(f"⚠️ {path.name} already exists. Open it and keep filling.")
        return

    text = TEMPLATE.format(country=country, today=date.today().isoformat())
    path.write_text(text, encoding="utf-8")
    print(f"✅ Created research file: {path}")

if __name__ == "__main__":
    main()