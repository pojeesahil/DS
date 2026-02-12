#TODO:1)make training and prediction file as seperate files
#TODO:2)find gov data to cross verify the soil erosion(currently we are just teaching it to mimic RULSE formula)
#TODO:3)maybe make a GUI where user can find soil erosion on diff location like google map
import ee
import geemap
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# you can choose your location here(will make it automatic later)
LAT = 19.93 
LON = 73.53
MY_PROJECT = 'gen-lang-client-0426799622' 
# Radius of area to analyze (in meters)
# 2750m radius = 5.5km x 5.5km box
#DO NOT INCREASE IT(it might exhaust our api)
SIZE_METERS = 2750 

# DEVICE SETUP (GPU vs CPU)
# This ensures the code runs on whatever hardware is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {device}")


def initialize_ee():
    try:
        #it initialize with the project ID
        print(f"Attempting to initialize with project: {MY_PROJECT}")
        ee.Initialize(project=MY_PROJECT)
    except Exception:
        # If it is your first time, it will try to log you in
        print("Authentication required. Opening browser...")
        ee.Authenticate()
        ee.Initialize(project=MY_PROJECT)
    print("✅ Earth Engine Initialized.")

initialize_ee()


print("Extracting satellite data and calculating factors...")
# Create Dynamic ROI (Box around your point)
point = ee.Geometry.Point([LON, LAT])
roi = point.buffer(SIZE_METERS).bounds()

# 1. Rainfall (R)
rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate('2023-01-01', '2023-12-31').sum().clip(roi)
R = rain.pow(1.61).multiply(0.0483).rename('R')

# 2. Soil Erodibility (K)
soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").clip(roi)
#[GEMINI Work,idk whats exactly is this either] Map USDA classes to K values (Sand=0.05 ... Clay=0.2 ... Silt=0.3)
K = soil.remap([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
               [0.05, 0.15, 0.2, 0.25, 0.3, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], 0.3).rename('K')

# 3. Slope (LS)
dem = ee.Image("USGS/SRTMGL1_003").clip(roi)
slope_deg = ee.Terrain.slope(dem)
LS = slope_deg.divide(100).pow(1.3).multiply(2).rename('LS')

# 4. Vegetation (C)
def mask_s2_clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
    return image.updateMask(mask).divide(10000)

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(roi)
      .filterDate('2023-01-01', '2023-06-30')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
      .map(mask_s2_clouds).median().clip(roi))

ndvi = s2.normalizedDifference(['B8', 'B4'])
C = ndvi.expression("exp(-2 * (ndvi / (1 - ndvi)))", {'ndvi': ndvi}).rename('C')
C = C.where(C.gt(1), 1).where(C.lt(0), 0)

# 5. Masking (Urban, Water, Snow)
landcover = ee.ImageCollection("ESA/WorldCover/v100").first().clip(roi)
valid_mask = landcover.neq(50).And(landcover.neq(80)).And(landcover.neq(100))
K_corrected = K.updateMask(valid_mask).unmask(0)

# 6. Soil Loss Calculation
soil_loss = R.multiply(K_corrected).multiply(LS).multiply(C)
class_map = ee.Image(0).where(soil_loss.gte(5).And(soil_loss.lt(20)), 1).where(soil_loss.gte(20), 2).rename('class')


inputs = s2.select(['B4', 'B3', 'B2', 'B8']).addBands(slope_deg.divide(90))
feature_stack = inputs.addBands(class_map)

print("Downloading pixels to NumPy... (this may take a minute)")
pixel_data = geemap.ee_to_numpy(feature_stack, region=roi, scale=30)
pixel_data = np.nan_to_num(pixel_data, nan=0.0)

# Convert to Tensors and move to Device (GPU or CPU)
data_tensor = torch.from_numpy(pixel_data).float().permute(2, 0, 1)
X = data_tensor[:5, :, :].unsqueeze(0).to(device)
Y = data_tensor[5, :, :].long().unsqueeze(0).to(device)

# THE ml MODEL
class MultiClassUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MultiClassUNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(32 + 16, 16)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(out_c))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        if d1.size() != e1.size(): d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat([d1, e1], dim=1)
        out = self.dec1(d1)
        return self.final(out)

model = MultiClassUNet(in_channels=5, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\nStarting Training...")
for epoch in range(101):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# Saving
print("Generating final map...")
with torch.no_grad():
    prediction = torch.argmax(model(X), dim=1).squeeze().cpu().numpy()

# Colourmap: Green, Yellow, Red
cmap = ListedColormap(['#228B22', '#FFD700', '#FF0000'])

plt.figure(figsize=(12, 6))

rgb = pixel_data[:, :, 0:3]
rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))

plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title(f"Satellite View ({LAT}, {LON})")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(prediction, cmap=cmap, vmin=0, vmax=2)
plt.title("AI Prediction (Green=Safe, Yellow=Mid, Red=High)")
plt.axis('off')

output_path = f"erosion_result_{LAT}_{LON}.png"
plt.savefig(output_path)
print(f"✅ Success! Results saved to: {output_path}")
plt.show() 