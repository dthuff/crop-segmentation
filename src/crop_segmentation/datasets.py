import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype


def create_dataset(config: dict, mode: str) -> Dataset:
    if mode not in ["train", "val", "test"]:
        raise AttributeError(f"Unexpected mode {mode} encountered in create_dataset()")
    img_dir = os.path.join(config["data"]["data_dir"], mode, "images", "rgb")
    label_dir = os.path.join(config["data"]["data_dir"], mode, "labels", config["model"]["target_class"])
    transform = create_transform_composition(config)
    dataset = RGBImageDataset(img_dir=img_dir, label_dir=label_dir, transform=transform, target_transform=transform)
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
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_list = glob(os.path.join(self.img_dir, '*.jpg'))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        _, f = os.path.split(img_path)
        label_path = os.path.join(self.label_dir, f.replace(".jpg", ".png"))
        image = Image.open(img_path)
        label = Image.open(label_path)  # Labels are png, images are jpg *shrug*
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
