import ee
import geemap
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# Location (change anytime)
LAT = 19.93
LON = 73.53

MY_PROJECT = "gen-lang-client-0426799622"
SIZE_METERS = 2750

MODEL_PATH = "models/erosion_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {device}")


def initialize_ee():
    try:
        print(f"Attempting to initialize with project: {MY_PROJECT}")
        ee.Initialize(project=MY_PROJECT)
    except Exception:
        print("Authentication required. Opening browser...")
        ee.Authenticate()
        ee.Initialize(project=MY_PROJECT)
    print("✅ Earth Engine Initialized.")


def get_prediction_data():
    print("Extracting satellite data...")

    point = ee.Geometry.Point([LON, LAT])
    roi = point.buffer(SIZE_METERS).bounds()

    # 1. Slope
    dem = ee.Image("USGS/SRTMGL1_003").clip(roi)
    slope_deg = ee.Terrain.slope(dem)

    # 2. Sentinel-2
    def mask_s2_clouds(image):
        qa = image.select("QA60")
        mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(mask).divide(10000)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate("2023-01-01", "2023-06-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_s2_clouds)
        .median()
        .clip(roi)
    )

    inputs = s2.select(["B4", "B3", "B2", "B8"]).addBands(slope_deg.divide(90))

    print("Downloading pixels to NumPy... (this may take a minute)")
    pixel_data = geemap.ee_to_numpy(inputs, region=roi, scale=30)
    pixel_data = np.nan_to_num(pixel_data, nan=0.0)

    data_tensor = torch.from_numpy(pixel_data).float().permute(2, 0, 1)
    X = data_tensor[:5, :, :].unsqueeze(0).to(device)

    return X, pixel_data


class MultiClassUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MultiClassUNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = self.conv_block(32 + 16, 16)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)

        if d1.size() != e1.size():
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:])

        d1 = torch.cat([d1, e1], dim=1)
        out = self.dec1(d1)
        return self.final(out)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Run train.py first.")

    model = MultiClassUNet(in_channels=5, num_classes=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"✅ Loaded model from {MODEL_PATH}")
    return model


def generate_map(prediction, pixel_data):
    cmap = ListedColormap(["#228B22", "#FFD700", "#FF0000"])

    plt.figure(figsize=(12, 6))

    rgb = pixel_data[:, :, 0:3]
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title(f"Satellite View ({LAT}, {LON})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(prediction, cmap=cmap, vmin=0, vmax=2)
    plt.title("AI Prediction (Green=Safe, Yellow=Mid, Red=High)")
    plt.axis("off")

    output_path = f"erosion_result_{LAT}_{LON}.png"
    plt.savefig(output_path)
    print(f"✅ Success! Results saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    initialize_ee()

    X, pixel_data = get_prediction_data()

    model = load_model()

    print("Generating prediction...")
    with torch.no_grad():
        outputs = model(X)
        prediction = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

    generate_map(prediction, pixel_data)
