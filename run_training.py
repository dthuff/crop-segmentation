import argparse
import os.path

import torch
from torch.utils.data import DataLoader

from crop_segmentation.datasets import create_dataset
from crop_segmentation.loss import DiceLoss2, DiceBCELoss
from crop_segmentation.model import UNet
from crop_segmentation.plotting import plot_and_save_loss
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

    train_dataset = create_dataset(config, "train")
    val_dataset = create_dataset(config, "val")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config["model"]["batch_size"], shuffle=True)

    # Init model and optimizer
    model = UNet(img_dim=config["model"]["img_dim"])
    model.to(device=torch.device(f'cuda:{cl_args.device}'))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["model"]["learning_rate"],
                                 weight_decay=config["model"]["weight_decay"])

    loss_dict = {"TRAIN_LOSS": [],
                 "VAL_LOSS": []}

    # Training loop
    for t in range(config["model"]["max_epochs"]):
        print(f"\n----\nEpoch {t}\n----")
        train_loss = train_loop(dataloader=train_dataloader,
                                model=model,
                                loss_fn=DiceBCELoss(),
                                optimizer=optimizer,
                                epoch_number=t,
                                save_dir=config["data"]["save_dir"])

        val_loss = val_loop(dataloader=val_dataloader,
                            model=model,
                            loss_fn=DiceBCELoss(),
                            epoch_number=t,
                            save_dir=config["data"]["save_dir"])

        loss_dict["TRAIN_LOSS"].append(train_loss)
        loss_dict["VAL_LOSS"].append(val_loss)

        plot_and_save_loss(loss_dict, config["data"]["save_dir"])
