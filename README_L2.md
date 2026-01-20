# TROPOMI CH4 (Methane) — L2 vs L3 Workflow README

This document explains how to find, download, clean, subset (CONUS), and process TROPOMI methane (CH4) data for the Level-2 (L2) product. The goal is to make it easy to reproduce this pipeline later (or scale it up to a longer time range).

---

## 1) Overview

### Products
- **L2 (Level-2):** swath-level retrievals (point-like observations), includes `qa_value` quality metric
- **L3 (Level-3):** gridded/aggregated product derived from L2 (often easier to use for analysis)

### What we did
- Downloaded L2 netCDF (`.nc`) files over selected dates
- Filtered L2 observations using QA thresholds
- Subset to CONUS to reduce size and speed up processing
- Exported cleaned data as CSV for analysis
- Compared QA value densities and distribution of L2 data

---

## 2) Repo Setup

### Clone repo
```bash
git clone https://github.com/erikylewis/methane_policy_evaluation.git
cd methane_policy_evaluation
```

### Pull latest
```bash
git pull
```

### Create environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3) Folder Structure (Recommended)

```
Projects/
  methane_policy_evaluation/
    full_l2_config.py
    l2_analysis.py
    tmp_run/
```

**Notes:**
- Store raw `.nc` files in `tmp_run` but are then deleted, while CSV file is kept

---

## 4) Key Variables

### L2 variables (typical)
- Methane: `CH4_column_volume_mixing_ratio_dry_air` 
- QA: `qa_value`
- Coordinates: `latitude`, `longitude` 

### QA filtering decision
We primarily used:
- Keep observations with `qa_value >= 0.4`
- Drop observations with `qa_value == 0` (typically no methane value reported)

---

## 5) Geographic Subsetting (CONUS)

To reduce data volume and focus comparisons, we subset to approximate CONUS:
- Latitude: **24 to 50**
- Longitude: **-125 to -66**

Example bounding box:

```python
CONUS_BBOX = {
  "lat_min": 24,
  "lat_max": 50,
  "lon_min": -125,
  "lon_max": -66
}
```
---

## 6) Step-by-Step Pipeline

### Step A — Download L2 Data and Export to CSV
**Script:** `full_l2_config.py`

**Purpose:**
- Download L2 `.nc` files for a specified date range
- `.nc` files save to: `methane_policy_evaluation/tmp_run` 
- Only include points specified by CONUS bounding box
- Points are exported to a `.csv` file
- `.nc` files are deleted keeping the folder `tmp_run` empty

**Example run:**
```bash
python full_l2_config.py \
--start YYYY-MM-DD --end YYYY-MM-DD \
--lat-min 24 --lat-max 50 \
--lon-min -125 --lon-max -66
```

### Step B — Compare L2 points with qa_value = 0.4 vs 1.0
**Script/Notebook:** `l2_analysis.py`

**Purpose:**
- Read in the generated csv of L2 files
- Generate kernal density and a smoothed histogram of ch4 distribution
- Compute binned LOWESS and plot QA value == 1 density

**Example run:**
```bash
python l2_analysis.py
```

---



## 7) Outputs

**Expected artifact:**
- Clean L2 CSV file: `/tmp_run/l2_YYYY-MM-DD_YYYY-MM-DD_ch4.csv`

---

## 8) Repro Notes / Decisions

- Methane values were not observed as true “zero” for L2 observations; there appears to be a baseline atmospheric concentration around ~1600
- Differences across `qa_value == 0.4` vs `qa_value == 1.0` did not appear strongly systematic in our sampled analysis.
- Determining whether observed differences are caused by QA interactions vs unobservables (e.g., clouds / measurement difficulty / geography) would require a larger-scale data collection effort and possibly joining with landcover or other environmental covariates.

---

## 9) Common Issues / Troubleshooting

### Credentials / Download Errors
- Some download sources require credentials or valid authentication tokens. The expired tokens used in past runs have been left in `full_l2_config.py`
- If AWS returns Forbidden (403), check permissions/credentials.
- Visit "https://documentation.dataspace.copernicus.eu/APIs/S3.html#generate-secrets" and click the link under 'Generate Secrets' to obtain your own secret keys to use to download L2 data

### Large Files / Performance
- L2 data is large (~40GB for 1 week of data over CONUS). CONUS subsetting is strongly recommended.

### Merge Conflicts
- If conflicts occur: stop and run `git status`, then ask for help before continuing.

---

## 10) Quick “Start Here” (Minimal)

- Install relevant libraries in virtual environment:

```python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install requests boto3 numpy pandas xarray netcdf4 h5netcdf matplotlib scipy statsmodels```

```bash
git pull
python Satalite_data/full_l2_config.py   # download L2 and export as CSV
python Satalite_data/l2_analysis.py   # analyze L2 and compare qa_values
```

---

## 11) Links / References
- GitHub issue thread with plots and notes: Issue #19
- Official documentation links (add): 'https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Methane-Product-Readme-File'