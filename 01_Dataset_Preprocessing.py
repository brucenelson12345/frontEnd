from glob import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from tqdm import tqdm


def make_folders(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    return output_path

def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format: 
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format: 
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """
    
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]

def convert_coco_json_to_yolo_txt(output_path, json_file):

    path = make_folders(output_path)

    with open(json_file) as f:
        json_data = json.load(f)

    # write _darknet.labels, which holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "_darknet.labels")
    with open(label_file, "w") as f:
        for category in tqdm(json_data["categories"], desc="Categories"):
            category_name = category["name"]
            f.write(f"{category_name}\n")

    for image in tqdm(json_data["images"], desc="Annotation txt for each iamge"):
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                category = anno["category_id"]
                bbox_COCO = anno["bbox"]
                x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("Converting COCO Json to YOLO txt finished!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="COCO Style Dataset Label Conversion: "
    "Generates the 'labels' associates with the 'images' dir. Also verifies annotations")

    parser.add_argument("-d", "--dataset_dir", dest="dataset_dir",
        help="path to a directory containing coco style .json and .yaml files")
    parser.add_argument("-c", "--config_dir", dest="config_dir",
        help="path to save new .json and .yaml files")

    args = parser.parse_args()

    # Verify dataset path exists
    if not os.path.exists(args.dataset_dir):
        print('Dataset directory path not found.')
        print('Quitting early.')
        quit()
    
    # Verify config path exists
    if not os.path.exists(args.config_dir):
        print('Config directory path not found.')
        print('Quitting early.')
        quit()
    
    # Generates labels for associated images called in the annotations .json
    for dir in os.scandir(args.dataset_dir):
        label_dir = args.dataset_dir + '/' + dir.name + '/labels'
        config_dir = args.config_dir + '/' + dir.name + '_coco.json'
        convert_coco_json_to_yolo_txt(label_dir, config_dir)
    
    # Verifies annotation categories from the annotations .json
    for file in os.listdir(args.config_dir):
        if file.endswith('.json'):
            f_config = open(args.config_dir + '/' + file) # => './datasets/images_thermal_val/coco.json'
            # returns JSON object as a dictionary
            config_val = json.load(f_config)
            # closing files
            f_config.close()

            categories = []

            for detection in config_val['annotations']:
                categories.append(detection['category_id'])

            print("{}: {}".format(file, np.unique(categories)))
