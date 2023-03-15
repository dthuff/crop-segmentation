import os
from glob import glob

from torch.utils.data import Dataset
from PIL import Image


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
