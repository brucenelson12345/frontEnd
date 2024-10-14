import json
import shutil
import os

file = 'config_person/images_rgb_train_coco.json'

with open(file, 'r') as openfile:
    data = json.load(openfile)

target_dir = "datasets/FLIR_ADAS_v2/images_rgb_train/images/"
dest_dir = "datasets/FLIR_ADAS_v2_person/images_rgb_train/images/"

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for images in data['images']:
    shutil.copyfile(target_dir + images["file_name"], dest_dir + images["file_name"])