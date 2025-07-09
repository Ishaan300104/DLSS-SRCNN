import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import kornia
import torch.nn.functional as F
from model import SimpleSRNet, DLSS_SRCNN

upscale_factor = 2
simple_sr_net_instance = SimpleSRNet(upscale_factor=upscale_factor)
model = DLSS_SRCNN(sr_model=simple_sr_net_instance, upscale_factor=upscale_factor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load weights and move model to device
model.load_state_dict(torch.load("best_weights.pth", map_location=device))
model.to(device) # Move the entire model to the selected device
model.eval() # Set the model to evaluation mode (important for BatchNorm, Dropout)

# Preprocessing
preprocessing_transform = transforms.Compose([
    transforms.ToTensor() # Scales to [0.0, 1.0] and converts to C, H, W
])

# Load and Preprocess Test Image
test_image_path = "DIV2K_valid_LR_960x540\\0803.png"
if not os.path.exists(test_image_path):
    print(f"Error: Test image not found at {test_image_path}")
    exit()

test_image = Image.open(test_image_path).convert('RGB')
test_image = preprocessing_transform(test_image) # test_image is now a FloatTensor in [0.0, 1.0], (C, H, W)

# Add batch dimension and move to device
test_image = test_image.unsqueeze(0).to(device) # (1, C, H, W)

# Performing Inference
with torch.no_grad():
    output = model(test_image) # output is (1, 3, H_HR, W_HR), likely float in [0.0, 1.0]

# Post-processing and Saving
output = output.squeeze(0).cpu().numpy() # output is now (C, H_HR, W_HR) as float numpy array

'''
Convert from C, H, W to H, W, C and scale to 0-255 for OpenCV
DLSS_SRCNN returns RGB, but OpenCV expects BGR. So, we need to permute channels from (C, H, W) to (H, W, C),
then convert RGB to BGR for OpenCV.
'''

output_bgr = (output * 255).astype("uint8") # Scale to 0-255 and convert to uint8
output_bgr = output_bgr.transpose(1, 2, 0) # Change to H, W, C
output_bgr = cv2.cvtColor(output_bgr, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV

output_dir = "output-images"
os.makedirs(output_dir, exist_ok=True)

# Define the output filename
output_filename = os.path.join(output_dir, f"output_803_upscaledddddd{upscale_factor}.png")

cv2.imwrite(filename=output_filename, img=output_bgr)
print(f"Saved image to: {output_filename}")