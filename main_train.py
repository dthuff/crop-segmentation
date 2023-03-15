import os.path
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype
from torch.utils.data import DataLoader

import torch
from dataload import RGBImageDataset
from model import UNet
from train import train_loop, val_loop
from loss import IoULoss

# Hyper parameters:
batch_size = 64
target_class = "weed_cluster"
channels = 3
img_dim = 128  # Must be factor of 16 (base UNet has 4 maxpools in encoder)
learning_rate = 0.001
max_epochs = 200
weight_decay = 5e-7
train_val_test_split = [0.8, 0.1, 0.1]  # Proportion of data for training, validation, and testing. Sums to 1
device = 'cuda'
resume = False  # Resume training from best_epoch.tar?
use_amp = False  # Use automatic mixed precision?

# Paths
data_dir = '/home/daniel/datasets/Agriculture-Vision-2021/'
save_dir = "./saved_models/"
if not os.path.exists(save_dir): os.makedirs(save_dir)

# Transform compose
transform_composition = Compose([
    ToTensor(),
    Resize(img_dim),
    ConvertImageDtype(torch.float)
])

# Dataset and loaders
train_dataset = RGBImageDataset(img_dir=os.path.join(data_dir, "train", "images", "rgb"),
                                label_dir=os.path.join(data_dir, "train", "labels", target_class),
                                transform=transform_composition,
                                target_transform=transform_composition)

val_dataset = RGBImageDataset(img_dir=os.path.join(data_dir, "val", "images", "rgb"),
                              label_dir=os.path.join(data_dir, "val", "labels", target_class),
                              transform=transform_composition,
                              target_transform=transform_composition
                              )

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True)

# Init model and optimizer
model = UNet(img_dim=img_dim)
model.to(device=device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)

start_epoch = 0

# Training loop
for t in range(start_epoch, max_epochs):
    print(f"Epoch {t}\n-------------------------------")
    train_loss = train_loop(dataloader=train_dataloader,
                            model=model,
                            loss_fn=IoULoss(),
                            optimizer=optimizer,
                            amp_on=use_amp)
