import numpy as np
import cv2
from PIL import Image
from model import SimpleSRNet, DLSS_SRCNN
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import math

def calculate_mse(original_image, processed_image):
        mse = np.mean((original_image - processed_image)**2)
        return mse

def calculate_psnr(mse, max_pixel_value):
        if mse == 0:
            return float('inf')
        psnr = 10 * math.log10((max_pixel_value**2) / mse)
        return psnr

image_number = 801
original_LR_file_path = f"DIV2K_valid_LR_960x540\\0{image_number}.png"
original_LR_image = Image.open(original_LR_file_path).convert(mode="RGB")
original_HR_file_path = f"DIV2K_valid_HR_1920x1080\\0{image_number}.png"
original_HR_image = Image.open(original_HR_file_path).convert(mode="RGB")

srnetinstance = SimpleSRNet(upscale_factor=2)
model = DLSS_SRCNN(sr_model=srnetinstance, upscale_factor=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load weights and move model to device
model.load_state_dict(torch.load("weights\\best_weights.pth", map_location=device))
model.to(device)
model.eval()

# Preprocessing
preprocessing_transform = transforms.Compose([
    transforms.ToTensor() # Scales to [0.0, 1.0] and converts to (C, H, W)
])

test_image = preprocessing_transform(original_LR_image)
test_image = test_image.unsqueeze(0).to(device) # (1, C, H, W)

# Performing Inference
with torch.no_grad():
    output = model(test_image)

# Post-processing
output = output.squeeze(0).cpu().numpy()
output_rgb = (output * 255).astype("uint8")
output_rgb = output_rgb.transpose(1, 2, 0) # Change to H, W, C
# output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

# Bicubic interpolation of LR image (for comparison)
bicubic_image = cv2.resize(src=cv2.imread(original_LR_file_path), dsize=(1920, 1080), interpolation=cv2.INTER_CUBIC)
bicubic_image = cv2.cvtColor(bicubic_image,cv2.COLOR_BGR2RGB)

# Calculating PSNR
max_pixel_value = 255

mse_value_srcnn = calculate_mse(original_HR_image, output_rgb)
psnr_value_srcnn = calculate_psnr(mse_value_srcnn, max_pixel_value)

mse_value_interpolated = calculate_mse(original_HR_image, bicubic_image)
psnr_value_interpolated = calculate_psnr(mse_value_interpolated, max_pixel_value)

print(f"PSNR (SRCNN): {psnr_value_srcnn:.2f} dB")
print(f"PSNR (Interpolation): {psnr_value_interpolated:.2f} dB")
print(f"MSE (SRCNN): {mse_value_srcnn:.2f}")
print(f"MSE (Interpolation): {mse_value_interpolated:.2f}")

# Plotting
size = (16, 9)
plt.figure(figsize=size)

plt.subplot(2, 2, 1)
plt.imshow(original_LR_image)
plt.title("Original LR image")

plt.subplot(2, 2, 2)
plt.imshow(original_HR_image)
plt.title("Original HR image")

plt.subplot(2, 2, 3)
plt.imshow(output_rgb)
plt.title(f"SRCNN (PSNR = {psnr_value_srcnn:.2f}, MSE = {mse_value_srcnn:.2f})")

plt.subplot(2, 2, 4)
plt.imshow(bicubic_image)
plt.title(f"Bicubic Interpolation (PSNR = {psnr_value_interpolated:.2f}, MSE = {mse_value_interpolated:.2f})")

plt.suptitle("Comparison between SRCNN and bicubic interpolation")
plt.tight_layout()
plt.show()