# Project: Hotel Booking Data Preprocessing (VoidMain Internship - Task 2)

## Context
- I'm Krishna, B.Tech AI & ML student, doing an AI internship at VoidMain Technologies.
- This is **Task 2: Data Analysis, Preprocessing & Feature Engineering** (70 pts, mandatory).
- Official brief: "Prepare real-world data for AI/ML applications. Select an appropriate public dataset. Perform data understanding, missing-value treatment, duplicate/outlier handling, encoding, scaling and suitable feature engineering/selection. Perform meaningful EDA and document preprocessing decisions."
- Submission: public GitHub repo + execution screenshots (uploaded to the repo) + submit the link on the internship portal. Optional: short video explaining the code + live run.
- This is a **standalone submission** with its own repo. Only Task 2 work goes here (no model training).
- Keep the output model-ready (clean data + saved preprocessor), but don't build anything beyond Task 2's scope.
- Environment: Windows, VS Code, Python. Project folder sits inside OneDrive Desktop - keep the virtual environment (`.venv`) out of git and avoid heavy generated folders.

## Dataset
- "Hotel booking demand" (Antonio, Almeida & Nunes, 2019, *Data in Brief*) - real bookings from two Portuguese hotels. Same data as the Kaggle version.
- **You download it yourself (no Kaggle login needed).** In Stage 0, if `data/raw/hotel_bookings.csv` doesn't exist, write a small script `src/download_data.py` that downloads it from the public TidyTuesday GitHub mirror:
  - Primary URL: `https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv`
  - Fallback URL (if primary fails): same path with `master` instead of `main`
  - Save as `data/raw/hotel_bookings.csv`
  - Verify: 119,390 rows and 32 columns; log a clear success/failure message
  - `main.py` should call this automatically if the raw file is missing, so the project runs from scratch on any machine
- Note: in this mirror, missing values in `agent`, `company`, `country` may appear as the string `"NULL"` instead of empty - treat `"NULL"` as missing (and mention this in decisions.md, it's a good real-world data point).
- Raw file at `data/raw/hotel_bookings.csv`. Never modify it.
- It's under GitHub's 100 MB limit, so commit it (needed for reproducibility and the deployed web app). Credit the source in the README.
- Target column (for context only): `is_canceled`

## Known data issues to handle
- Missing: `company` (mostly empty), `agent`, `country`, `children` - each needs its own justified treatment
- Many duplicate rows
- `adr` has negative and extreme values; some bookings have 0 total guests (invalid)
- High-cardinality `country`
- Leakage columns for cancellation prediction: `reservation_status`, `reservation_status_date` - flag and drop from the model-ready output (explain why)

## Brief requirement -> where it lives (keep this mapping true)
| Requirement | Code | Shown in |
|---|---|---|
| Public dataset selection | README "Dataset" section | README |
| Data understanding | `src/load_data.py`, notebook section 1 | notebook |
| Missing-value treatment | `src/cleaning.py` | notebook + decisions.md |
| Duplicate handling | `src/cleaning.py` | notebook + decisions.md |
| Outlier handling | `src/cleaning.py` | notebook + decisions.md + figures |
| Encoding | `src/pipeline.py` | notebook + decisions.md |
| Scaling | `src/pipeline.py` | notebook + decisions.md |
| Feature engineering | `src/features.py` | notebook + decisions.md |
| Feature selection | `src/features.py` / notebook | notebook + decisions.md |
| Meaningful EDA | notebook + `reports/figures/` | notebook + README insights |
| Documented decisions | `docs/decisions.md` | README summary |

## Project structure
```
data/raw/            # original CSV
data/processed/      # cleaned + model-ready output
notebooks/eda.ipynb  # main walkthrough (run top to bottom with outputs saved)
models/              # preprocessor.joblib
tests/               # pytest tests
src/
  config.py          # paths, URLs, column lists, thresholds, random_state
  download_data.py   # downloads + verifies the raw dataset
  load_data.py
  cleaning.py        # missing, duplicates, outliers, invalid rows
  features.py        # feature engineering + selection helpers
  pipeline.py        # encoding + scaling (ColumnTransformer/Pipeline)
  visualize.py       # shared plotting functions + style
app/                 # Flask web app (app.py, templates/, static/)
reports/figures/     # saved plots
screenshots/         # execution screenshots for submission
docs/decisions.md    # preprocessing decisions log
main.py              # runs full pipeline end to end
README.md
requirements.txt
.gitignore
```

## Stage roadmap (do ONE stage, then STOP and wait for me)
0. **Setup** - create ALL folders and starter files yourself, `.venv`, `requirements.txt` (pinned), `.gitignore`, `src/config.py`, `src/download_data.py` (download + verify dataset), `git init`
1. **Data understanding** - shape, dtypes, info, describe, missing %, unique counts, first observations
2. **Cleaning** - missing values, duplicates, invalid rows, outliers (+ tests)
3. **Feature engineering** - new features (+ tests)
4. **EDA** - meaningful plots with written insights, saved to `reports/figures/`
5. **Encoding + scaling** - sklearn ColumnTransformer/Pipeline, train/test split, fit on train only, save `models/preprocessor.joblib`
6. **Feature selection** - correlation/redundancy/leakage checks, final feature list
7. **Integration** - `main.py` runs everything raw -> processed in one command; notebook runs top to bottom cleanly
8. **Quality pass** - all tests pass, ruff clean, docstrings/type hints everywhere, `docs/decisions.md` complete
9. **README** - professional README (overview, dataset + credit, pipeline diagram, how to run, EDA insights with figures, decisions summary, structure, tech stack, live app link, author)
10. **Web app (NO Streamlit)** - my own custom web app: Flask backend + hand-built HTML/CSS/JS frontend with a polished, professional design (not a template look). Pages: dataset overview, before/after cleaning stats, interactive EDA charts (Chart.js or Plotly.js), and upload-a-CSV -> download preprocessed output. Reuses `src/` + saved pipeline, no duplicated logic. We'll plan the design together before building.
11. **Screenshots** - help me capture: `main.py` run, pytest passing, notebook key outputs, web app pages. Save to `screenshots/` and link them in the README.
12. **GitHub** - guide me to create a public repo, add remote, push (I'll handle login). Check nothing unwanted is committed.
13. **Deploy live (not localhost)** - deploy the Flask app to a free public host (e.g. Render) with `gunicorn` + a proper start command; add the live link to the README.
14. **Video prep** - short talking-points script: problem, dataset, each preprocessing step + why, live run of `main.py` and the app.
15. **Final check** - verify every row of the requirement mapping table above is satisfied; then I submit the repo link on the portal.

## Engineering standards (this must look professional, NOT vibe-coded)
- **Type hints + docstrings** on every function (Google style). Small, single-purpose functions.
- **No hardcoded paths/magic numbers** - paths, column lists, thresholds, random_state in `src/config.py`.
- **Logging, not random prints** - `logging` module with a clean format for step-by-step progress (before/after counts), so terminal output looks good in screenshots.
- **Reproducible** - fixed `random_state`, pinned versions, `python main.py` goes from raw CSV to processed output in one command.
- **Tests** - pytest for cleaning and feature functions (small fake DataFrames). All tests must pass before a stage is done.
- **Formatting/linting** - `ruff format` + `ruff check`. No unused imports, no dead code.
- **Validation** - functions check inputs (required columns exist, no unexpected nulls after cleaning) and raise clear errors.
- **No data leakage** - split before fitting scalers/encoders; drop leaky columns.
- **Consistent plots** - one shared style, proper titles/labels, saved at good resolution.
- **Git** - one clean commit per stage with clear messages (e.g. `feat: add missing-value treatment`, `docs: add EDA insights`). Never commit `.venv`, caches, or secrets.
- I must **understand every line** - after each stage, give a short explanation of what was done and why, so I can explain it in the video.

## How to work with me
- One stage at a time. Show results, then STOP and wait for my go-ahead.
- Before big changes, briefly tell me the plan first.
- Use pandas, numpy, matplotlib/seaborn, scikit-learn, joblib, pytest, ruff, flask, gunicorn. NO Streamlit. No unnecessary libraries.
- Print/log clear before/after numbers at each step.
- Explain things casually and briefly; I prefer direct, practical answers.
