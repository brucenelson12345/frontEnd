import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts selected dataset to "
    "COCO style format. Flir datset will need the directory called `data` "
    "renamed to `images` as what Yolo expects it to be in order to run  train,"
    "validation, and test. Annotations images paths in json files will remove "
    "the data directory paths.")

    parser.add_argument("-i", "--input_dir", dest="input_dir",
        help="path to flir directory to convert to coco style format")

    args = parser.parse_args()

    # Verify dataset path exists
    if not os.path.exists(args.input_dir):
        print('Input directory path not found.')
        print('Quitting early.')
        quit()

    for dir in os.scandir(args.input_dir):
        if dir.is_dir():
            # Renames directories with images from data to images
            if os.path.isdir(dir.path + "/data"):
                os.rename(dir.path + "/data", dir.path + "/images")

            print(dir.path + '/' + 'coco.json')
            with open(dir.path + '/' + 'coco.json', 'r') as file:
                data = json.load(file)
            
            # Replace the previous data directory path with images
            for image in data["images"]:
                image["file_name"] = image["file_name"].replace("data/", "")
                image["subdirs"] = "images"
            
            with open(dir.path + '/' + 'coco.json', 'w') as file:
                json.dump(data, file, indent=2)
