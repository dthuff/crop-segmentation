import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype


def create_dataset(config: dict, mode: str) -> Dataset:
    if mode not in ["train", "val", "test"]:
        raise AttributeError(f"Unexpected mode {mode} encountered in create_dataset()")
    else:
        transform = create_transform_composition(config)

        match mode:
            case "train":
                img_dir = config["data"]["train_image_dir"]
                label_dir = os.path.join(config["data"]["train_labels_dir"], config["data"]["target_class"])
                dataset = RGBImageDataset(img_dir=img_dir, label_dir=label_dir, transform=transform)
            case "val":
                img_dir = config["data"]["val_image_dir"]
                label_dir = os.path.join(config["data"]["val_labels_dir"], config["data"]["target_class"])
                dataset = RGBImageDataset(img_dir=img_dir, label_dir=label_dir, transform=transform)
            case _:
                dataset = []

    return dataset


def create_transform_composition(config):
    # Transform compose
    transform_composition = Compose([
        ToTensor(),
        Resize(config["model"]["img_dim"], antialias=True),
        ConvertImageDtype(torch.float)
    ])
    return transform_composition


class RGBImageDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, target_transform=None):
        self.label_dir = label_dir
        self.label_list = glob(os.path.join(self.label_dir, '*.png'))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.label_list[idx])
        image_path = os.path.join(self.img_dir, os.path.basename(label_path).replace(".png", ".jpg"))

        label = Image.open(label_path)
        image = Image.open(image_path)

        if self.transform:
            label = self.transform(label)
            image = self.transform(image)

        return image, label
