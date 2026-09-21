"""
Clean HECAT standards by removing contaminated expectation rows,
then overwrite the original TSV.

TO RUN:
python ./src/clean_hecat_standards.py
"""

import shutil
import pandas as pd

INPUT_PATH = "data/cleaned/hecat_standards.tsv"
BACKUP_PATH = "data/cleaned/hecat_standards_backup.tsv"
MAX_EXPECTATION_LENGTH = 300

def main():
    df = pd.read_csv(INPUT_PATH, sep="\t")
    print(f"Original rows: {len(df)}")

    bad_mask = (
        (df["expectation"].str.len() > MAX_EXPECTATION_LENGTH) |
        df["expectation"].str.contains("Knowledge Expectations Grades", case=False, na=False)
    )

    bad = df[bad_mask]
    clean = df[~bad_mask].copy()
    clean["expectation"] = clean["expectation"].str.strip()

    print(f"Removing:      {len(bad)}")
    print(f"Remaining:     {len(clean)}")

    print("\nRemoved rows:")
    print(bad[["code", "grade_code", "module_code", "expectation"]].to_string(index=False))

    shutil.copy2(INPUT_PATH, BACKUP_PATH)
    clean.to_csv(INPUT_PATH, sep="\t", index=False)

    print(f"\nBackup: {BACKUP_PATH}")
    print(f"Cleaned: {INPUT_PATH}")

if __name__ == "__main__":
    main()