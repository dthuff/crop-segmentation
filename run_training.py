import argparse
import os.path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype

from crop_segmentation.dataload import RGBImageDataset
from crop_segmentation.loss import IoULoss
from crop_segmentation.model import UNet
from crop_segmentation.train import train_loop, val_loop, parse_training_config


def parse_cl_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config")
    arg_parser.add_argument("--device")
    return arg_parser.parse_args()


if __name__ == "__main__":
    cl_args = parse_cl_args()
    config = parse_training_config(cl_args.config)

    if not os.path.exists(config["data"]["save_dir"]):
        os.makedirs(config["data"]["save_dir"])

    # Transform compose
    transform_composition = Compose([
        ToTensor(),
        Resize(config["model"]["img_dim"], antialias=True),
        ConvertImageDtype(torch.float)
    ])

    # Dataset and loaders
    train_dataset = RGBImageDataset(img_dir=os.path.join(config["data"]["data_dir"], "train", "images", "rgb"),
                                    label_dir=os.path.join(config["data"]["data_dir"], "train", "labels", config["model"]["target_class"]),
                                    transform=transform_composition,
                                    target_transform=transform_composition)

    val_dataset = RGBImageDataset(img_dir=os.path.join(config["data"]["data_dir"], "val", "images", "rgb"),
                                  label_dir=os.path.join(config["data"]["data_dir"], "val", "labels", config["model"]["target_class"]),
                                  transform=transform_composition,
                                  target_transform=transform_composition
                                  )

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config["model"]["batch_size"],
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config["model"]["batch_size"],
                                shuffle=True)

    # Init model and optimizer
    model = UNet(img_dim=config["model"]["img_dim"])
    model.to(device=torch.device(f'cuda:{cl_args.device}'))

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["model"]["learning_rate"],
                                 weight_decay=config["model"]["weight_decay"])

    start_epoch = 0

    # Training loop
    for t in range(start_epoch, config["model"]["max_epochs"]):
        print(f"Epoch {t}\n-------------------------------")
        train_loss = train_loop(dataloader=train_dataloader,
                                model=model,
                                loss_fn=IoULoss(),
                                optimizer=optimizer,
                                amp_on=config["model"]["use_amp"])

        val_loss = val_loop(dataloader=val_dataloader,
                            model=model,
                            loss_fn=IoULoss(),
                            epoch_number=t)
