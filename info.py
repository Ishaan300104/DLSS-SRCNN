from model import DLSS_SRCNN, SimpleSRNet
import torch
from torchinfo import summary
from config import batch_size

'''
This file is just for printing the model summary (including #parameters)
'''

# Model    
SRNet_instance = SimpleSRNet(upscale_factor=2)
model = DLSS_SRCNN(sr_model=SRNet_instance, upscale_factor=2)
x = torch.randn(batch_size, 3, 960, 540) # we want 64 (batch_size = 64) images with 3 channels and dim of image is 960 x 540
print(f"Total number of parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
summary(model, input_size=(batch_size, 3, 960, 540))