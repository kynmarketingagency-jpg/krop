# check_data_table.py
# 👩‍🏫 Shows each CSV as a clean table.
# It finds the /data folder relative to THIS file, so it works from any folder.
# If pandas isn't installed, it falls back to the built-in csv module.

import os
from pathlib import Path

# 1) Locate the data folder relative to this script
ROOT = Path(__file__).resolve().parent           # folder that contains this .py file
DATA = ROOT / "data"                             # .../krop/data

files = [
    "countries.csv",
    "crops.csv",
    "reg_policies.csv",
    "studies.csv",
    "mapping.csv",
]

# 2) Try to use pandas for pretty tables, else fall back to csv
try:
    import pandas as pd
    use_pandas = True
except Exception:
    use_pandas = False
    import csv

print(f"🔎 Looking in: {DATA}\n")

for name in files:
    path = DATA / name
    print(f"📂 {name}:")
    if not path.exists():
        print(f"   ⚠️ Not found: {path}\n")
        continue

    try:
        if use_pandas:
            df = pd.read_csv(path)
            # Print as a nice table without row numbers
            print(df.to_string(index=False))
        else:
            # Minimal table if pandas isn't available
            with open(path, newline="", encoding="utf-8") as f:
                reader = list(csv.reader(f))
            if not reader:
                print("   (empty)")
            else:
                # Compute simple column widths
                widths = [max(len(cell) for cell in col) for col in zip(*reader)]
                for row in reader:
                    line = " | ".join(cell.ljust(w) for cell, w in zip(row, widths))
                    print("   " + line)
        print()  # blank line after each table
    except Exception as e:
        print(f"   ⚠️ Could not read {name}: {e}\n")