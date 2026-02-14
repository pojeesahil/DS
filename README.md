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

Run the analysis using:
Bash
`python main.py`

## Output Interpretation
The analysis generates a dual-pane visualization. The first pane displays the Sentinel-2 RGB satellite composite, while the second pane displays the AI-generated risk map.

The risk map follows a three-tier classification:

Green (Stable): Minimal soil loss (less than 5 tons/hectare/year).

Yellow (Moderate): Areas requiring monitoring (5 to 20 tons/hectare/year).

Red (Critical): High-risk zones requiring immediate intervention (greater than 20 tons/hectare/year).

Non-soil areas such as urban centers and water bodies are automatically masked and displayed as null values to prevent statistical bias.

## Soil Erosion Detection using RUSLE and Deep LearningT
his repository provides a quantitative approach to identifying soil erosion risk by integrating the Revised Universal Soil Loss Equation (RUSLE) with satellite imagery and neural networks. 
By leveraging Google Earth Engine for environmental data and PyTorch for spatial analysis, this system classifies terrain into distinct risk zones.

### Overview 
The project operates by calculating a scientific ground truth for soil loss based on five environmental factors. This data is then used to train a U-Net architecture—a convolutional neural network designed for semantic segmentation. Once trained, the model can identify high-risk areas from multispectral satellite data and topographic slope, providing an automated alternative to manual physics-based calculations.Technical MethodologyThe system quantifies soil loss ($A$) using the standard RUSLE formula:$$A = R*K*LS*C*P   

R(Rainfall Erosivity): Derived from CHIRPS daily precipitation data to measure the kinetic energy of rainfall

K (Soil Erodibility): Calculated from OpenLandMap soil texture classes, representing the susceptibility of soil particles to detachment.

LS (Slope Length and Steepness): Generated from NASA’s SRTM Digital Elevation Model to account for topographic effects on runoff velocity.

C (Cover Management): Calculated using the Normalized Difference Vegetation Index (NDVI) from Sentinel-2 imagery to evaluate how vegetation protects the soil surface.

P (Support Practices): Evaluated as a constant in this implementation, assuming natural land management.To ensure accuracy, the model incorporates an environmental mask using the ESA WorldCover dataset to exclude non-soil surfaces such as urban infrastructure, permanent water bodies, and snow.
## Run Instructions

### Train the model
python train.py

### Predict erosion map
python predict.py

### Run using main menu
python main.py
