# setup_files.py
# 👩‍🏫 Hey student! This script is like a helper robot.
# Its job is to build empty "bookshelves" (CSV files) for our project.
# Each CSV will be a shelf where we later put facts about GMOs.

import os
import csv

# Step 1: Define the folder where we want to store data
data_folder = os.path.join("krop", "data")

# Step 2: Make sure the folder exists (if not, create it)
os.makedirs(data_folder, exist_ok=True)

# Step 3: Define the files and their headers
files = {
    "countries.csv": ["country", "region", "gmo_stance", "notes", "source"],
    "crops.csv": ["crop", "trait", "description", "source"],
    "reg_policies.csv": ["country", "crop", "policy_type", "decision", "year", "notes", "source"],
    "studies.csv": ["study_id", "crop", "focus", "findings", "year", "source"],
    "mapping.csv": ["country", "crop", "prevalence", "notes", "source"]
}

# Step 4: Create each CSV file with only the header row
for filename, headers in files.items():
    filepath = os.path.join(data_folder, filename)
    # Open the file in write mode
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # Write the header row
    print(f"Created {filename} with headers: {headers}")

print("✅ All CSV files are ready in the 'krop/data' folder!")
