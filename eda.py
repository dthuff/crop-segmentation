# Exploratory data analysis
import os
from PIL import Image
from glob import glob

data_dir = "/home/daniel/Projects/data/Agriculture-Vision-2021/train/"

im_list = glob(data_dir + "images/rgb/*.jpg")

# We have 56K tiles
print(len(im_list))

# Each tile is of size (512, 512)
sizes = []

for im_path in im_list[::500]:
    im = Image.open(im_path)
    sizes.append(im.size)

#
label_classes = glob(data_dir + "labels/*/")
print(label_classes)

for im_path in im_list[::500]:
    
    im = Image.open(im_path)
    for label_path in label_classes:
        label_list = glob(label_path + ".png")
        for label_im in label_list[::500]:
            lb = Image.open(lb)
