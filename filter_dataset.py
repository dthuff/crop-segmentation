# This dataset has high prevalence of class imbalance. E.g., class "storm_damage" is present in only ~300/60000 (0.5%)
# of training images. So, I might want to enrich the % of positive samples that some stage of the pipeline sees.
# To do this, I want to filter down to sets of images that all contain a given target class.
import os
import torch

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype

data_dir = "/home/daniel/datasets/Agriculture-Vision-2021/val/labels"
data_dir_filtered = data_dir + "_filtered"

classes = os.listdir(data_dir)

transform_composition = Compose([
    ToTensor(),
    Resize(128, antialias=True),
    ConvertImageDtype(torch.float)
])

for img_class in classes:
    label_dir_this_class = f"{data_dir}/{img_class}/"
    if not os.path.isdir(os.path.join(data_dir_filtered, img_class)):
        os.makedirs(os.path.join(data_dir_filtered, img_class))
    for lb in os.listdir(label_dir_this_class):
        label_path = os.path.join(label_dir_this_class, lb)
        label = Image.open(label_path)
        if transform_composition(label).sum():
            label.save(os.path.join(data_dir_filtered, img_class, lb))
