# s2-lfmc-thailand
s2-lfmc-thailand: Sentinel-2–Based Heuristic Framework for Near-Real-Time Live Fuel Moisture Content Mapping in Northern Thailand
s2-lfmc-thailand provides an open, reproducible implementation of a Sentinel-2–based workflow for estimating relative vegetation moisture conditions (Live Fuel Moisture Content, LFMC) across northern Thailand. The workflow follows the methodology described in the manuscript:

Chotamonsak et al. (2025). “Towards Near Real-Time Estimation of Live Fuel Moisture Content from Sentinel-2 for Fire Management in Northern Thailand.”

This repository includes the full Python code used for preprocessing, spectral-index computation, normalization, moisture proxy integration, and heuristic LFMC estimation. It is intended to support transparency, reproducibility, and future development of a validated LFMC monitoring system for fire management applications in mainland Southeast Asia.
# s2-lfmc-thailand

Near-real-time **Live Fuel Moisture Content (LFMC)** mapping for northern Thailand using **Sentinel-2**.

This repository provides a single pipeline script:

- `lfmc_pipeline.py`

which:

- reads Sentinel-2 data (from **Sentinel Hub** or **local GeoTIFFs**),
- computes LFMC (%) using NDVI, NDII, MSI and an NDVI-based ETf proxy,
- generates **daily** LFMC maps and danger-class maps,
- aggregates LFMC to **weekly / biweekly / monthly** composites,
- exports all products as **PNG** maps and optional **GeoTIFFs**.

> Designed for operational fire management and seasonal dryness monitoring over northern Thailand, but easily adaptable to other regions.

---

## 1. Main features (LFMC only)

- **Daily LFMC grids**
  - Per-pixel LFMC (%) for a user-defined AOI and period.
  - Outputs:
    - PNG maps (`lfmc_YYYY-MM-DD.png`)
    - optional GeoTIFFs (`lfmc_YYYY-MM-DD.tif`)

- **LFMC danger classes**
  - Pixel-wise danger classes based on LFMC thresholds.
  - Outputs:
    - PNG + PDF map + histogram (`lfmc_danger_YYYY-MM-DD.png/.pdf`)
    - optional GeoTIFFs (`lfmc_danger_YYYY-MM-DD.tif`)

- **Temporal composites**
  - Weekly (`--agg weekly`)
  - Biweekly (`--agg biweekly`)
  - Monthly (`--agg monthly`)
  - Statistic: `--agg_stat mean` or `--agg_stat median`
  - Composite LFMC maps + danger-class maps for each period.

- **Flexible data sources**
  - `--s2_source sh` – directly from Sentinel Hub.
  - `--s2_source local` – local B04/B08/B11/SCL GeoTIFF stacks.

- **Boundary overlays**
  - Optional shapefile/GeoJSON overlay clipped to AOI.
  - Automatic province labels when suitable fields exist.

- **Parallel per-day processing**
  - `--workers N` with safe `spawn` start method.
  - Uses on-disk `.npy` grids to keep memory usage reasonable.

---

## 2. Installation

### 2.1. Python environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows

