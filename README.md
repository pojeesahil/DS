# Installation and Setup
The system is designed to run in a local Python environment.

## 1. Clone the Repository
Download the source code to your local machine:
go to your vscode terminal and type
Bash
```
git clone https://github.com/pojeesahil/DS.git
cd DS
```
## 2. Environment Configuration
Install the necessary dependencies using the provided requirements file:
Bash
```
pip install -r requirements.txt
```

install jupyter notebook extension in vscode preferably

**Requirements:**

earthengine-api: Google Earth Engine interface.

geemap: Geospatial analysis and visualization.

torch: PyTorch framework for neural network training.

numpy: Numerical data processing.

matplotlib: Results visualization and mapping.

## 3. Google Earth Engine Authentication
The project requires access to Google’s Earth Engine API. Upon the first execution of the script, a browser window will open to authorize the application.

Log in with a Google account associated with a registered Earth Engine project.

Select the appropriate Cloud Project ID when prompted.

Run the analysis using either:
1)manually running predict.ipynb in vscode
2)Running predict.ipynb on google collab in broswer

# Soil Erosion Detection using RUSLE, Time-Series Satellite Data, and Deep Learning

This repository provides a quantitative approach to identifying soil erosion risk by integrating the **Revised Universal Soil Loss Equation (RUSLE)** with multi-year time-series satellite imagery and deep learning. By leveraging Google Earth Engine for environmental data and PyTorch for spatial analysis, this system evaluates terrain to classify genuine landscape degradation risks.

## Overview

The project operates on a hybrid "Theory plus Reality" model. It calculates a scientific baseline for soil loss based on environmental factors (theoretical risk) and cross-references this with a five-year historical analysis of actual vegetation loss (observed reality). This dual-verification data trains a **U-Net convolutional neural network** to identify high-risk areas from multispectral satellite data and topographic slope, filtering out seasonal noise.

## Technical Methodology

The system quantifies soil loss ($A$) using the standard RUSLE formula:

$$A = R * K * LS * C * P$$

* **R (Rainfall Erosivity):** Derived from CHIRPS daily precipitation data to measure the kinetic energy of rainfall.
* **K (Soil Erodibility):** Calculated from OpenLandMap soil texture classes, representing soil particle detachment susceptibility.
* **LS (Slope Length and Steepness):** Generated from the USGS SRTM Digital Elevation Model to account for topographic effects on runoff velocity.
* **C (Cover Management):** Calculated using NDVI from Sentinel-2 imagery to evaluate surface protection by vegetation.
* **P (Support Practices):** Evaluated as a constant, assuming natural land management.

### Split-Logic Processing
To ensure accuracy across diverse landscapes, the model employs specialized logic based on the **ESA WorldCover** dataset:
* **Wildlands and Forests:** Classification requires both a severe mathematical RUSLE score and a significant historical drop in NDVI (10%+).
* **Agricultural Lands:** To avoid false alarms caused by standard crop harvesting cycles, farmlands are evaluated strictly on physical topography and rainfall thresholds.
* **Masking:** Urban infrastructure, permanent water bodies, and snow are excluded to prevent statistical bias.

## Output Interpretation

The analysis generates a three-pane visualization:
1.  **Satellite View:** A true-color RGB composite of the selected region.
2.  **Theoretical RUSLE Baseline:** The raw mathematical calculation highlighting all vulnerable slopes.
3.  **Hybrid AI Inference:** The finalized neural network prediction pinpointing actual, confirmed degradation.

### Risk Classification
* **Green (Stable):** Minimal calculated soil loss or stable vegetation cover.
* **Yellow (Vulnerable):** High-tension mathematical buffer zones acting as a warning tier.
* **Red (Critical):** High-risk zones requiring immediate intervention, exhibiting both topographical vulnerability and proven historical degradation.

### Level 3 Mitigation Advisory
The system includes a targeted intervention feature. When critical (Red) zones are detected, the model:
* Identifies the largest contiguous erosion cluster using center-of-mass calculations.
* Drops a dynamic GPS marker on the interactive map.
* Provides engineering recommendations (e.g., Gabion walls for steep slopes vs. riparian buffers for low slopes).