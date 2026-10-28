# Preprocessing Decisions Log

Every preprocessing choice made in this project, why it was made, and what the
alternative would have been. Updated stage by stage.

## 1. Data loading

### 1.1 `"NULL"` strings are treated as missing values
- **What:** `pd.read_csv(..., na_values=["NULL"])` (token list lives in `src/config.py`).
- **Why:** The TidyTuesday mirror writes missing values in `agent`, `company`
  and `country` as the literal text `"NULL"` instead of an empty cell. Pandas
  would otherwise read them as a real category called `"NULL"` (or keep the
  `agent`/`company` columns as text instead of numbers), hiding over 100k
  missing values. This is a common real-world trap: "missing" is not always
  blank.
- **Note:** `children` uses `"NA"`, which pandas already recognises as missing by default.

### 1.2 Raw data is never modified
- **What:** `data/raw/hotel_bookings.csv` is read-only; all outputs go to `data/processed/`.
- **Why:** Reproducibility - anyone can rerun `python main.py` from the original file.
  The downloader writes to a temporary `.part` file and only renames it after the
  shape check (119,390 x 32) passes, so a broken download can never replace it.
