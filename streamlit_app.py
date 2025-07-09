import streamlit as st
from model import DLSS_SRCNN, SimpleSRNet
from PIL import Image
from torchvision import transforms
import torch
import time
import io # Import io for BytesIO

st.title("Ultraweight DLSS-SRCNN")
st.subheader("Enter your low resolution image that you want to upscale to 1920x1080")

# Placeholder for the upsampled image display
upsampled_image_placeholder = st.empty()

# Model and device
srnetinstance = SimpleSRNet(upscale_factor=2)
model = DLSS_SRCNN(sr_model=srnetinstance, upscale_factor=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load weights and move model to device
model.load_state_dict(torch.load("best_weights.pth", map_location=device))
model.to(device)
model.eval()

global_upsampled_pil_image = None

def run_upsampling_process():
    global global_upsampled_pil_image # Declare intent to modify the global variable

    # Access the uploaded file from the module-level variable
    if uploaded_image_file is None:
        st.warning("Please upload an image file first.")
        return

    # Open the image using PIL (Pillow) to handle various image formats
    pil_image = Image.open(uploaded_image_file)

    # Define the preprocessing transformation
    preprocessing_transform = transforms.Compose([
        transforms.Resize((540, 960)),  # Resize to Height, Width as expected by model input
        transforms.ToTensor()          # Convert PIL Image/numpy.ndarray to FloatTensor in [0.0, 1.0]
    ])

    # Apply the transformation to the PIL Image
    input_tensor = preprocessing_transform(pil_image) # input_tensor is now a FloatTensor (C, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(device) # Add batch dimension (1, C, H, W)

    st.text("Upsampling the image, please wait a few seconds...")

    bar = st.progress(0) # Start from 0 for the progress bar

    # Performing Inference
    with torch.no_grad():
        output = model(input_tensor) # output is (1, 3, H_HR, W_HR), likely float in [0.0, 1.0]

    # bar.progress(100) # Complete the progress bar

    # Post-processing
    output = output.squeeze(0).cpu().numpy() # Remove batch dim, move to CPU, convert to numpy (C, H_HR, W_HR)
    output_rgb = (output * 255).astype("uint8") # Scale to 0-255 and convert to uint8
    output_rgb = output_rgb.transpose(1, 2, 0) # Change from (C, H, W) to (H, W, C) for displaying

    # Convert numpy array back to PIL Image for display and download
    global_upsampled_pil_image = Image.fromarray(output_rgb)

    # Display the upsampled image using the placeholder
    with upsampled_image_placeholder:
        st.image(global_upsampled_pil_image, caption="Upsampled Image", use_container_width=False)


with st.sidebar:
    st.header("Input image") # Removed width=400 as it's not a valid argument for header
    uploaded_image_file = st.file_uploader("Input image file (Note: image will be resized to match 16:9 aspect ratio)") # Removed width=400 as it's not a valid argument for file_uploader

    # Store the button state in session_state to re-run download button logic
    if st.button("Click here to upsample", on_click=run_upsampling_process, use_container_width=True):
        st.session_state['upsample_done'] = True
    else:
        # Reset state if button is not clicked (e.g., on initial load or rerun without click)
        st.session_state['upsample_done'] = False


# Download button logic - only show if upsampling has been done and an image exists
if 'upsample_done' in st.session_state and st.session_state['upsample_done'] and global_upsampled_pil_image is not None:
    # Convert PIL Image to bytes
    buf = io.BytesIO()
    # Save as PNG to the buffer. You can change format if needed (e.g., 'JPEG')
    global_upsampled_pil_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Click here to download high resolution image",
        data=byte_im,
        file_name="upscaled_image.png",
        mime="image/png",
        use_container_width=True
    )
else:
    # You might want to display a message or nothing if no image is ready
    st.info("Upload an image and click 'Upsample' to generate the high-resolution image for download.")