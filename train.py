import torch
from model import SimpleSRNet, DLSS_SRCNN
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import SRDataset
from torch import optim
from config import learning_rate, weight_decay, batch_size, num_epochs, dropout_rate
import torch.nn as nn
from loss_function import loss_fn
import matplotlib.pyplot as plt

# Data directories
TRAIN_HR_DIR = 'DIV2K_train_HR_1920x1080'
TRAIN_LR_DIR = 'DIV2K_train_LR_960x540'
VAL_HR_DIR = 'DIV2K_valid_HR_1920x1080'
VAL_LR_DIR = 'DIV2K_valid_LR_960x540'

# Define your target sizes for LR and HR images.
LR_HEIGHT, LR_WIDTH = 540, 960
HR_HEIGHT, HR_WIDTH = LR_HEIGHT * 2, LR_WIDTH * 2

lr_transforms = transforms.Compose([
    transforms.Resize((LR_HEIGHT, LR_WIDTH)), # Resize all LR images to a fixed size
    transforms.ToTensor()
])
hr_transforms = transforms.Compose([
    transforms.Resize((HR_HEIGHT, HR_WIDTH)), # Resize all HR images to a fixed size
    transforms.ToTensor()
])

# Dataset instances
train_dataset = SRDataset(hr_root_dir=TRAIN_HR_DIR, lr_root_dir=TRAIN_LR_DIR, transform_hr=hr_transforms, transform_lr=lr_transforms)
val_dataset = SRDataset(hr_root_dir=VAL_HR_DIR, lr_root_dir=VAL_LR_DIR, transform_hr=hr_transforms, transform_lr=lr_transforms)

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Device
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

# Initializing network
srmodel_instance = SimpleSRNet(upscale_factor=2)
model = DLSS_SRCNN(sr_model=srmodel_instance, upscale_factor=2).to(device)

# Loss and Optimizer
loss_type = loss_fn
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# Training the SRCNN:----------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculating loss for training set
train_losses = [] # This will store train loss after each epoch

def train(epoch):
    train_loss=0
    for batch_index, (lr_image, hr_image) in enumerate(train_dataloader):

        # get data to cuda if possible
        lr_image = lr_image.to(device=device)
        hr_image = hr_image.to(device=device)

        # forward
        output = model(lr_image)
        loss = loss_type(output, hr_image)

        # backward
        optimizer.zero_grad() #setting gradients to zero
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        train_loss += loss.item()
    train_loss = train_loss/len(train_dataloader)

    # storing the train losses in a list
    train_losses.append(train_loss)



# calculating loss for validation set
validation_losses = []

def val(epoch):
    validation_loss=0
    with torch.no_grad():
        for batch_index, (lr_image, hr_image) in enumerate(val_dataloader):

            # get data to cuda if possible
            lr_image = lr_image.to(device=device)
            hr_image = hr_image.to(device=device)

            output = model(lr_image)
            loss = loss_type(output, hr_image)

            validation_loss += loss.item()
    validation_loss /= len(val_dataloader)

    # storing the validation losses in a list
    validation_losses.append(validation_loss)
    
best_val_loss = 10e6
# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    train(epoch)

    if (validation_losses[epoch] < best_val_loss):
        torch.save(model.state_dict(), "best_weights.pth")
        best_val_loss = validation_losses[epoch]

    model.eval()
    val(epoch)
    print(f"Train loss = {train_losses[epoch]}")
    print(f"Val loss = {validation_losses[epoch]}")
    learning_rate = learning_rate*(0.99)**epoch

# Plotting graphs
plt.plot(range(1, num_epochs+1), train_losses, '-o')
plt.plot(range(1, num_epochs+1), validation_losses, '-o')
plt.xlabel('# epoch')
plt.ylabel('losses')
plt.legend(['Training','Validation'])
plt.title('Training and validation losses')
plt.show()