import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype
from torchvision.transforms import v2


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
    transforms = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.RandomResizedCrop(size=(config["model"]["img_dim"], config["model"]["img_dim"]), antialias=True),
        v2.RandomAffine(degrees=180, translate=(0.3, 0.3)),
        ConvertImageDtype(torch.float)
    ])
    return transforms


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

        label = ToTensor()(label)
        image = ToTensor()(image)

        if self.transform:
            label_image_stack = torch.concatenate((label, image), dim=0)
            label_image_stack = self.transform(label_image_stack)
            label = label_image_stack[0, :, :].unsqueeze_(0)
            image = label_image_stack[1:, :, :]

        return image, label
