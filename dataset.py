from torch.utils.data import Dataset
from PIL import Image
import os

class SRDataset(Dataset):
    def __init__(self, hr_root_dir, lr_root_dir, transform_hr=None, transform_lr=None):
        self.hr_root_dir = hr_root_dir
        self.lr_root_dir = lr_root_dir
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr

        # List of HR image filenames (LR files have same names)
        self.image_filenames = sorted([
            file for file in os.listdir(hr_root_dir)
            if os.path.isfile(os.path.join(hr_root_dir, file)) and file.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        filename = self.image_filenames[index]

        # Full paths for HR and LR images
        hr_img_path = os.path.join(self.hr_root_dir, filename)
        lr_img_path = os.path.join(self.lr_root_dir, filename)

        # Loading images
        # hr_image = Image.open(hr_img_path)
        # lr_image = Image.open(lr_img_path)
        hr_image = Image.open(hr_img_path).convert('RGB')
        lr_image = Image.open(lr_img_path).convert('RGB')

        # Applying transformations
        if self.transform_lr:
            lr_image = self.transform_lr(lr_image)
        if self.transform_hr:
            hr_image = self.transform_hr(hr_image)

        # lr_image (input feature), hr_image (output label)
        return lr_image, hr_image
