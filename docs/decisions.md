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

## 2. Cleaning (`src/cleaning.py`)

Steps run in this order: invalid rows -> duplicates -> missing values -> category
fixes -> outlier capping. Rule values live in `src/config.py`.

**Result: 119,390 -> 87,208 rows (27.0% removed), 129,425 missing cells -> 0.**

| Step | Rows removed | Values changed |
|---|---:|---:|
| Drop zero-guest bookings | 180 | - |
| Drop negative ADR | 1 | - |
| Drop implausible counts | 21 | - |
| Drop exact duplicates | 31,980 | - |
| `company` -> `has_company` flag | - | column replaced |
| Fill `agent` + `has_agent` flag | - | 12,128 |
| Fill `country` "Unknown" | - | 447 |
| Fill `children` 0 | - | 4 |
| `meal` Undefined -> SC | - | 492 |
| Cap extreme ADR | - | 85 |

(Fill counts are lower than the raw missing counts because duplicates were removed first.)

### 2.1 Order of steps: duplicates are removed before any column is recoded
- **Why:** If `company` were dropped first, rows that differed *only* in their company ID
  would suddenly look identical and be wrongly deleted (7 rows in this data).
  Invalid rows go first so the counts match the raw data exactly (180 zero-guest rows).
- **Side effect:** 4 rows look duplicated *after* cleaning. They were different bookings
  in the raw file (different company IDs) and are kept on purpose.

### 2.2 `company` (94.3% missing) -> `has_company` flag, column dropped
- **Why:** With 94% empty and 352 different IDs, the ID itself carries almost no usable
  signal and would explode into hundreds of sparse one-hot columns. Missing here means
  "no company paid for this booking", which is information worth keeping as a 0/1 flag.
- **Alternative considered:** Keep the ID and group rare companies into "Other". Rejected:
  still ~94% one category, adds complexity for little signal.

### 2.3 `agent` (13.7% missing) -> fill 0 + `has_agent` flag
- **Why:** Missing means "booked without a travel agent", not unknown. Agent IDs start
  at 1, so 0 is a safe "no agent" code; the flag makes "direct vs agent" explicit for models.
  The ID column is kept (86% filled, real signal) and will be treated as a category later.
- **Alternative considered:** Fill with the most common agent (mode). Rejected: it would
  invent a fake relationship with agent #9 for 12k direct bookings.

### 2.4 `country` (0.4% missing) -> "Unknown"
- **Why:** An explicit category keeps the rows and lets a model learn if "unknown origin"
  behaves differently. Too few rows to matter either way, so the honest label wins.
- **Alternative considered:** Fill with the mode (PRT). Rejected: PRT guests cancel far
  more than average (56.6% vs 37.0% raw), so guessing PRT could bias the data.

### 2.5 `children` (4 rows missing) -> 0
- **Why:** 4 rows out of 119k; 0 is by far the most common value (93%) and these
  bookings already have adults. Also restores `children` to an integer column.
- **Alternative considered:** Drop the 4 rows. Equally valid; filling keeps them.

### 2.6 `meal` "Undefined" -> "SC"
- **Why:** The dataset paper defines both "Undefined" and "SC" as "no meal package".
  Two labels for one meaning would split the same group in two.

### 2.7 Zero-guest bookings (180 rows) -> dropped
- **Why:** 0 adults + 0 children + 0 babies is not a real stay; these are system or
  entry errors and would distort per-guest features.

### 2.8 Exact duplicates (31,980 rows) -> dropped
- **Trade-off:** There is no booking ID, so some duplicates are probably genuine identical
  bookings: 46% are in the "Groups" segment, 42% are Non-Refund deposits, and one block
  has 180 identical rows. These look like group blocks entered one room at a time.
- **Why drop anyway:** Identical rows that end up on both sides of a train/test split
  let a model "memorise" test answers, giving inflated scores (leakage). They also
  over-weight a few large group blocks.
- **Effect to be aware of:** The duplicates are 63% canceled, so the cancellation rate
  drops from **37.0% to 27.5%**. The cleaned data describes *unique booking patterns*,
  not raw booking volume - any cancellation-rate figure should say which one it uses.
- **Alternative considered:** Keep duplicates and use a group-aware split so copies never
  cross train/test, or keep one row plus a "copies" count column. More faithful to volume
  but more complex; dropping is the standard, leakage-safe choice for this task.

### 2.9 Implausible counts (21 rows) -> dropped, using a gap rule (not IQR)
- **Rule:** For each count column, the maximum is the highest value seen in a meaningful
  number of bookings, right before a clear gap in the frequency table:

  | Column | Max kept | Values seen above it |
  |---|---:|---|
  | `adults` | 4 | 5, 6, 10, 20, 26, 27, 40, 50, 55 (16 rows) |
  | `children` | 3 | 10 (1 row) |
  | `babies` | 2 | 9, 10 (2 rows) |
  | `required_car_parking_spaces` | 3 | 8 (2 rows) |

- **Why:** Every booking with 10+ adults is a canceled Resort booking with ADR 0 -
  group placeholders, not real rooms. 9-10 babies or 8 parking spaces for 2 adults are typos.
  Dropped rather than capped because the true value is unknown (55 could have meant 5 or 2).
- **Why not IQR:** Count columns are mostly 0 or 2, so their IQR is exactly 0. The IQR rule
  would flag every booking with 1+ child (8,590), 1+ parking space (7,416), 1+ baby (917)
  or 3+ adults (6,280) - thousands of normal families - as outliers.

### 2.10 ADR (average daily rate)
- **Negative (1 row, -6.38) -> dropped.** A price cannot be negative.
- **Zero (1,959 raw rows) -> kept.** All 715 zero-night (day-use) bookings have ADR 0, and
  most "Complementary" segment bookings do too - these are real free stays, not errors.
- **Extreme high -> capped at the 99.9th percentile (335.00), 85 rows affected.**
  The max of 5,400 is 10x the next value (510) for a 2-adult, 1-night standard room - clearly
  an error - while prices up to ~500 are plausible for premium rooms in peak season.
  Capping (winsorising) keeps the rows but stops one value from dominating means and scaling.
- **Alternative considered:** IQR fence (Q3 + 1.5 x IQR = 211). Rejected: it would cap
  3,793 rows, mostly genuine peak-season prices.
- **Leakage note:** The cap is computed on the full cleaned data before the train/test split.
  A 99.9th percentile barely moves with which rows are held out, so the effect is negligible;
  the stricter alternative is to fit the cap on the training set only inside the pipeline.

### 2.11 Leakage columns are flagged, not dropped yet
- **What:** `reservation_status` and `reservation_status_date` stay in the cleaned file and
  a warning is logged. They are removed in feature selection (Stage 6).
- **Why:** `reservation_status` maps one-to-one onto the target and its date is recorded
  after the outcome, so they must never reach a model - but they are still useful for EDA
  (e.g. checking the leakage itself).

### 2.12 Validation
- `clean_data` checks all required columns exist before running, and raises an error if
  any missing value survives cleaning.
