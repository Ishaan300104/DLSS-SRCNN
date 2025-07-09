import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

class SimpleSRNet(nn.Module):
    """A simple CNN for single-channel super-resolution."""
    def __init__(self, upscale_factor=2):
        super(SimpleSRNet, self).__init__()
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x


class DLSS_SRCNN(nn.Module):
    """
    Wraps the Y-channel SR network to handle the full YCbCr workflow.
    This is the model we train directly.
    """
    def __init__(self, sr_model, upscale_factor=2):
        super(DLSS_SRCNN, self).__init__()
        self.sr_model = sr_model
        self.upscale_factor = upscale_factor

    def forward(self, x_rgb):
        
        # Convert RGB to YCbCr
        x_ycbcr = kornia.color.rgb_to_ycbcr(x_rgb)

        # Split channels
        y_channel, cb_channel, cr_channel = torch.chunk(x_ycbcr, chunks=3, dim=1)

        # Apply the SR model to the Y channel
        sr_y_channel = self.sr_model(y_channel)

        # Upscale color channels with simple bicubic interpolation
        target_height, target_width = sr_y_channel.shape[2:]
        upscaled_cb = F.interpolate(cb_channel, size=(target_height, target_width), mode='bicubic', align_corners=False)
        upscaled_cr = F.interpolate(cr_channel, size=(target_height, target_width), mode='bicubic', align_corners=False)

        # Combine the super-resolved Y with the upscaled Cb and Cr
        sr_ycbcr = torch.cat([sr_y_channel, upscaled_cb, upscaled_cr], dim=1)

        # Convert back to RGB and clamp the output to the valid [0, 1] range
        sr_rgb = kornia.color.ycbcr_to_rgb(sr_ycbcr)
        return torch.clamp(sr_rgb, 0.0, 1.0)