# check_data.py
# 👩‍🏫 This script is like opening each bookshelf (CSV file)
# and reading the books (rows) to see what's inside.

import os
import csv

# Step 1: Point to the data folder
data_folder = os.path.join("krop", "data")

# Step 2: List of CSV files we want to check
files = [
    "countries.csv",
    "crops.csv",
    "reg_policies.csv",
    "studies.csv",
    "mapping.csv"
]

# Step 3: Loop through each file and print contents
for filename in files:
    filepath = os.path.join(data_folder, filename)
    print(f"\n📂 {filename}:")
    # Open the file
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            print("   ", row)  # Print each row as a list