import ee
import geemap
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import concurrent.futures
import os

# ==========================================
# PROJECT SETUP
# ==========================================
MY_PROJECT = "gen-lang-client-0426799622"
SIZE_METERS = 2750  # Creates a approximately 5.5km x 5.5km bounding box
MODEL_PATH = "models/erosion_model_hybrid.pth"
os.makedirs("models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"System: Initializing neural network on {device}")

# ==========================================
# BALANCED TRAINING DATASET
# Mix of stable regions and critical erosion hotspots
# ==========================================
LOCATIONS = [
    # STABLE REGIONS
    (19.93, 73.53),  # Nashik
    (21.14, 79.08),  # Nagpur 
    (16.99, 73.30),  # Ratnagiri
    (20.93, 77.75),  # Amravati 
    (19.87, 75.34),  # Chhatrapati Sambhajinagar 
    
    # EROSION HOTSPOTS
    (19.16, 73.68),  # Malin Village 
    (18.08, 73.42),  # Mahad 
    (17.92, 73.56),  # Ambenali Ghat 
    (19.33, 73.76),  # Malshej Ghat 
    (17.38, 73.74),  # Patan / Koyna Catchment 
]

def initialize_ee():
    try:
        ee.Initialize(project=MY_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=MY_PROJECT)
    print("System: Earth Engine Initialized.\n")

def get_hybrid_training_data_batch():
    print("Network: Fetching spatial data for 10 locations simultaneously. Please wait...")

    X_list, Y_unified_list = [], []

    def mask_s2_clouds(image):
        qa = image.select('QA60')
        mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(mask).divide(10000)

    def process_location(lat_lon_tuple):
        lat, lon = lat_lon_tuple
        try:
            point = ee.Geometry.Point([lon, lat])
            roi = point.buffer(SIZE_METERS).bounds()

            # 1. SMART MASKING
            landcover = ee.ImageCollection("ESA/WorldCover/v100").first().clip(roi)
            valid_mask = landcover.neq(50).And(landcover.neq(80)).And(landcover.neq(100))
            
            is_farm = landcover.eq(40) 
            is_wild = landcover.neq(40).And(valid_mask) 

            # 2. THE GEOGRAPHY
            dem = ee.Image("USGS/SRTMGL1_003").clip(roi)
            slope_deg = ee.Terrain.slope(dem)
            LS = slope_deg.divide(100).pow(1.3).multiply(2).rename('LS') 

            soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").clip(roi)
            K = soil.remap([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                           [0.05, 0.15, 0.2, 0.25, 0.3, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], 0.3).rename('K')
            K_corrected = K.updateMask(valid_mask).unmask(0)

            # 3. THE 2019 BASELINE
            s2_2019 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                       .filterBounds(roi).filterDate('2019-01-01', '2019-06-30')
                       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                       .map(mask_s2_clouds).median().clip(roi))
            
            rain_2019 = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate('2019-01-01', '2019-12-31').sum().clip(roi)
            R_2019 = rain_2019.pow(1.61).multiply(0.0483).rename('R') 
            
            ndvi_2019 = s2_2019.normalizedDifference(['B8', 'B4'])
            C_2019 = ndvi_2019.expression("exp(-2 * (ndvi / (1 - ndvi)))", {'ndvi': ndvi_2019}).rename('C')
            C_2019 = C_2019.where(C_2019.gt(1), 1).where(C_2019.lt(0), 0)

            soil_loss_2019 = R_2019.multiply(K_corrected).multiply(LS).multiply(C_2019)
            
            # 4. THEORETICAL RUSLE (Razor-Thin Yellow, Expanded Red)
            # Yellow band drastically squeezed to a 5-point margin (110-115).
            # Red ceiling lowered to 115 to capture more high-risk terrain.
            y_rusle = ee.Image(0).where(soil_loss_2019.gte(110).And(soil_loss_2019.lt(115)), 1) \
                                 .where(soil_loss_2019.gte(115), 2)

            # 5. THE REALITY CHECK (2024 Data)
            s2_2024 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                       .filterBounds(roi).filterDate('2024-01-01', '2024-06-30')
                       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                       .map(mask_s2_clouds).median().clip(roi))
            ndvi_2024 = s2_2024.normalizedDifference(['B8', 'B4'])
            ndvi_loss = ndvi_2019.subtract(ndvi_2024)

            # 6. THE UNIFIED LABEL
            y_unified = ee.Image(0)
            
            # Base Vulnerability
            y_unified = y_unified.where(y_rusle.gte(1), 1) 
            
            # Wildland Rule (Math high + relaxed 6% vegetation drop)
            y_unified = y_unified.where(is_wild.And(y_rusle.eq(2)).And(ndvi_loss.gt(0.06)), 2)
            
            # Farmland Rule (Catastrophic math threshold lowered to 115)
            y_unified = y_unified.where(is_farm.And(soil_loss_2019.gt(115)), 2)
            
            y_unified = y_unified.updateMask(valid_mask).unmask(0).rename('y_unified')

            # 7. STACK & DOWNLOAD
            inputs = s2_2019.select(['B4', 'B3', 'B2', 'B8']).addBands(slope_deg.divide(90))
            feature_stack = inputs.addBands(y_unified)

            pixel_data = geemap.ee_to_numpy(feature_stack, region=roi, scale=30)
            pixel_data = np.nan_to_num(pixel_data, nan=0.0)

            data_tensor = torch.from_numpy(pixel_data).float().permute(2, 0, 1).unsqueeze(0) 
            data_tensor = F.interpolate(data_tensor, size=(192, 192), mode='nearest')
            
            print(f"Network: Successfully processed region {lat}, {lon}")
            return data_tensor
            
        except Exception as e:
            print(f"Network: Error processing region {lat}, {lon} - {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_location, LOCATIONS))

    for tensor in results:
        if tensor is not None:
            X_list.append(tensor[:, :5, :, :])
            Y_unified_list.append(tensor[:, 5, :, :].long())

    X_batch = torch.cat(X_list, dim=0).to(device)         
    Y_unified_batch = torch.cat(Y_unified_list, dim=0).to(device)

    unique, counts = torch.unique(Y_unified_batch, return_counts=True)
    pixel_counts = dict(zip(unique.tolist(), counts.tolist()))
    print("\nSystem: PIXEL DISTRIBUTION IN TRAINING DATA:")
    print(f"  Green (Stable): {pixel_counts.get(0, 0)} pixels")
    print(f"  Yellow (Vulnerable): {pixel_counts.get(1, 0)} pixels")
    print(f"  Red (Critical): {pixel_counts.get(2, 0)} pixels")

    return X_batch, Y_unified_batch.squeeze(1)

# ==========================================
# U-NET ARCHITECTURE 
# ==========================================
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
            nn.ReLU(), nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        if d1.size() != e1.size(): d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat([d1, e1], dim=1)
        out = self.dec1(d1)
        return self.final(out)

# ==========================================
# FOCAL LOSS FUNCTION
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==========================================
# TRAINING EXECUTION
# ==========================================
def train_unified_model(X, Y_unified):
    model = MultiClassUNet(in_channels=5, num_classes=3).to(device)
    
    # Adjusted weights: Added a 1.5x bias to Red to ensure confident classification 
    # of danger zones now that the mathematical floor is significantly higher.
    weights = torch.tensor([1.0, 0.5, 2], dtype=torch.float).to(device)
    
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("\nSystem: Commencing model training...")
    for epoch in range(151): 
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y_unified)
        loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0:
            print(f"  Epoch {epoch} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSystem: Model weights saved successfully to {MODEL_PATH}")

if __name__ == "__main__":
    initialize_ee()
    X, Y_unified = get_hybrid_training_data_batch()
    train_unified_model(X, Y_unified)