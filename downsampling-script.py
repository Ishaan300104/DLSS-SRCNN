# This script downsamples the high resolution images to half its dimension
'''
I have used bilinear downsampling to downscale DIV2K HR images.
If you are running this, you can experiment with various other downsampling methods
'''

import cv2
import os
from tqdm import tqdm

# Paths
train_folder_path = "DIV2K_train_HR"
val_folder_path = "DIV2K_valid_HR"

resized_train_folder_path = "DIV2K_train_HR_1920x1080"
resized_val_folder_path = "DIV2K_valid_HR_1920x1080"

output_train_folder_path = "DIV2K_train_LR_960x540"
output_val_folder_path = "DIV2K_valid_LR_960x540"


# Creating new folders for resized and downscaled images
os.makedirs(name=resized_train_folder_path, exist_ok=True)
os.makedirs(name=resized_val_folder_path, exist_ok=True)
os.makedirs(name=output_train_folder_path, exist_ok=True)
os.makedirs(name=output_val_folder_path, exist_ok=True)
num_unprocessed_images = 0

print(f"\nWorking with training images...\n")

# Working with training data
for index in tqdm(range(1, 801)):

    # Processing "DIV2K_train_HR" images
    file_path = f"{train_folder_path}/{"0" * (4 - len(str(index)))}{index}.png"
    image = cv2.imread(file_path)
    
    # Resizing train images to 1920x1080 (screen aspect ratio) and saving it
    resized_image = cv2.resize(src=image, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)
    resized_path = f"{resized_train_folder_path}/{"0" * (4 - len(str(index)))}{index}.png"
    
    if not os.path.exists(resized_path):
        cv2.imwrite(resized_path, resized_image)
        num_unprocessed_images += 1
    else:
        # print(f"File {resized_path} already exists!")
        pass

    # Downscaling dimensions
    new_width = resized_image.shape[1] // 2
    new_height = resized_image.shape[0] // 2
    new_dimensions = (new_width, new_height)
    
    # Bilinear downsampling and saving images
    downsampled_image = cv2.resize(src=resized_image, dsize=new_dimensions, interpolation=cv2.INTER_LINEAR)
    downsampled_path = f"{output_train_folder_path}/{"0" * (4 - len(str(index)))}{index}.png"
    
    if not os.path.exists(downsampled_path):
        cv2.imwrite(downsampled_path, downsampled_image)
        num_unprocessed_images += 1
    else:
        # print(f"File {downsampled_path} already exists!")
        pass

if num_unprocessed_images > 0:
    print(f"\nProcessed images from: {train_folder_path}")
    print(f"Saved resized images to: {resized_train_folder_path}")
    print(f"Saved downsampled images to: {output_train_folder_path}\n")
else:
    print("\n Images already exist, please check respective folders.")



num_unprocessed_images = 0
print(f"\nWorking with validation images...\n")

# Working with validation data
for index in tqdm(range(801, 901)):

    # Processing "DIV2K_valid_HR" images
    file_path = f"{val_folder_path}/{"0" * (4 - len(str(index)))}{index}.png"
    image = cv2.imread(file_path)
    
    # Resizing train images to 1920x1080 (screen aspect ratio) and saving it
    resized_image = cv2.resize(src=image, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)
    resized_path = f"{resized_val_folder_path}/{"0" * (4 - len(str(index)))}{index}.png"

    if not os.path.exists(resized_path):
        cv2.imwrite(resized_path, resized_image)
        num_unprocessed_images += 1

    else:
        # print(f"File {resized_path} already exists!")
        pass

    # Downscaling dimensions
    new_width = resized_image.shape[1] // 2
    new_height = resized_image.shape[0] // 2
    new_dimensions = (new_width, new_height)
    
    # Bilinear downsampling and saving images
    downsampled_image = cv2.resize(src=resized_image, dsize=new_dimensions, interpolation=cv2.INTER_LINEAR)
    downsampled_path = f"{output_val_folder_path}/{"0" * (4 - len(str(index)))}{index}.png"
    
    if not os.path.exists(downsampled_path):
        cv2.imwrite(downsampled_path, downsampled_image)
        num_unprocessed_images += 1
    else:
        # print(f"File {downsampled_path} already exists!")
        pass


if num_unprocessed_images > 0:
    print(f"\nProcessed images from: {val_folder_path}")
    print(f"Saved resized images to: {resized_val_folder_path}")
    print(f"Saved downsampled images to: {output_val_folder_path}\n")
else:
    print("\n Images already exist, please check respective folders.")



























# cv2.imshow("Downscaled image", downsampled_image)
# cv2.waitKey(0)