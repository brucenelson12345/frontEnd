import json
from pathlib import Path
import os
import yaml
import shutil


class CocoFilter():
    """ Filters the COCO dataset
    """
    def _process_info(self):
        self.info = self.coco['info']
        
    def _process_licenses(self):
        self.licenses = self.coco['licenses']
        
    def _process_categories(self):
        self.categories = dict()
        self.super_categories = dict()
        self.category_set = set()

        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']
            
            # Add category to categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
                self.category_set.add(category['name'])
            else:
                print(f'ERROR: Skipping duplicate category id: {category}')
            
            # Add category id to the super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {cat_id} # e.g. {1, 2, 3} |= {4} => {1, 2, 3, 4}

    def _process_images(self):
        self.images = dict()
        for image in self.coco['images']:
            image_id = image['id']
            if image_id not in self.images:
                self.images[image_id] = image
            else:
                print(f'ERROR: Skipping duplicate image id: {image}')
                
    def _process_segmentations(self):
        self.segmentations = dict()
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

    def _filter_categories(self):
        """ Find category ids matching args
            Create mapping from original category id to new category id
            Create new collection of categories
        """
        missing_categories = set(self.filter_categories) - self.category_set
        if len(missing_categories) > 0:
            print(f'Did not find categories: {missing_categories}')
            should_continue = input('Continue? (y/n) ').lower()
            if should_continue != 'y' and should_continue != 'yes':
                print('Quitting early.')
                quit()

        self.new_category_map = dict()
        new_id = 1
        for key, item in self.categories.items():
            if item['name'] in self.filter_categories:
                self.new_category_map[key] = new_id
                new_id += 1

        self.new_categories = []
        for original_cat_id, new_id in self.new_category_map.items():
            new_category = dict(self.categories[original_cat_id])
            new_category['id'] = new_id
            self.new_categories.append(new_category)

    def _filter_annotations(self):
        """ Create new collection of annotations matching category ids
            Keep track of image ids matching annotations
        """
        self.new_segmentations = []
        self.new_image_ids = set()
        anno_id = 1
        for image_id, segmentation_list in self.segmentations.items():
            for segmentation in segmentation_list:
                original_seg_cat = segmentation['category_id']
                if original_seg_cat in self.new_category_map.keys():
                    new_segmentation = dict(segmentation)
                    new_segmentation['category_id'] = self.new_category_map[original_seg_cat] - 1
                    new_segmentation['id'] = anno_id
                    anno_id += 1
                    self.new_segmentations.append(new_segmentation)
                    self.new_image_ids.add(image_id)

    def _filter_images(self):
        """ Create new collection of images
        """
        self.new_images = []
        self.new_anno_image_id = {}
        new_image_id = 0
        for image_id in self.new_image_ids:
            self.new_anno_image_id[image_id] = new_image_id
            new_image_id += 1
            self.new_images.append(self.images[image_id])
    
    def _update_annotations_image_ids(self):
        """ Map new image ids to new annotations
        """
        for image in self.new_images:
            image["id"] = self.new_anno_image_id[image["id"]]
        for segmentation in self.new_segmentations:
            segmentation["image_id"] = self.new_anno_image_id[segmentation["image_id"]]

    def filter_json(self, input_json, output_json, categories):
        # Open json
        self.input_json_path = Path(input_json)
        self.output_json_path = Path(output_json)
        self.filter_categories = categories

        # Verify input path exists
        if not self.input_json_path.exists():
            print('Input json path not found.')
            print('Quitting early.')
            quit()
        
        # Load the json
        print('Loading json file...')
        with open(self.input_json_path) as json_file:
            self.coco = json.load(json_file)
        
        # Process the json
        print('Processing input json...')
        self._process_info()
        self._process_licenses()
        self._process_categories()
        self._process_images()
        self._process_segmentations()

        # Filter to specific categories
        print('Filtering...')
        self._filter_categories()
        self._filter_annotations()
        self._filter_images()
        self._update_annotations_image_ids()

        # Build new JSON
        new_master_json = {
            'info': self.info,
            'licenses': self.licenses,
            'images': self.new_images,
            'annotations': self.new_segmentations,
            'categories': self.new_categories
        }

        # Write the JSON to a file
        print('Saving new json file...')
        with open(self.output_json_path, 'w+') as output_file:
            json.dump(new_master_json, output_file, indent=2)

        print('Filtered json saved.')

    def filter_yaml(self, input_yaml, output_yaml, dataset_dir, categories):
        with open(input_yaml) as stream:
            try:
                input_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        input_data['nc'] = len(categories)
        input_data['names'] = categories
        dataset_types = ["train", "val", "test"]
        for dataset_type in dataset_types:
            parts = Path(input_data[dataset_type]).parts
            stripped_path = Path(*parts[parts.index("datasets") + 1:])
            input_data[dataset_type] = dataset_dir + '/' + str(stripped_path)
        
        with open(output_yaml, 'w') as outfile:
            yaml.dump(input_data, outfile, default_flow_style=None)
    
    def image_sorter(self, config_file, dataset_input, dataset_output):
        with open(config_file, 'r') as openfile:
            data = json.load(openfile)

        dataset_type = (config_file.rsplit('/', 1)[-1]).rsplit('_', 1)[0]
        input_dir = dataset_input + '/' + dataset_type + '/images/'
        output_dir = dataset_output + '/' + dataset_type + '/images/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for images in data['images']:
            shutil.copyfile(input_dir + images["file_name"], output_dir + images["file_name"])
    
    def parse_output_path(self, input_dir, output_dir, categories):
        # Config file output dir, if none given, output dir is input dir + categories given
        if output_dir is None:
            separator = '_'
            new_output_dir = input_dir + '_' + separator.join(categories)
        else:
            new_output_dir = output_dir
        
        # Verify config output path does not already exist
        if os.path.exists(new_output_dir):
            should_continue = input('Config output path already exists. Overwrite? (y/n) ').lower()
            if should_continue != 'y' and should_continue != 'yes':
                print('Quitting early.')
                quit()
        else:
            os.mkdir(new_output_dir)
            print(new_output_dir)
            return new_output_dir

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter COCO JSON: "
    "Filters a COCO Instances JSON file to only include specified categories. "
    "This includes images, and annotations. Does not modify 'info' or 'licenses'.")
    
    parser.add_argument("-ci", "--config_input_dir", dest="config_input_dir",
        help="path to a directory containing coco style .json and .yaml files")
    parser.add_argument("-co", "--config_output_dir", dest="config_output_dir",
        help="path to save new .json and .yaml files")
    parser.add_argument("-di", "--dataset_input_dir", dest="dataset_input_dir",
        help="path dataset root of different image sets.")
    parser.add_argument("-do", "--dataset_output_dir", dest="dataset_output_dir",
        help="path to save new filtered dataset images.")
    parser.add_argument("-c", "--categories", nargs='+', dest="categories",
        help="List of category names separated by spaces, e.g. -c person dog bicycle")

    args = parser.parse_args()

    # Verify input path exists
    if not os.path.exists(args.config_input_dir):
        print('Input directory path not found.')
        print('Quitting early.')
        quit()
    if not os.path.exists(args.dataset_input_dir):
        print('Dataset directory path not found.')
        print('Quitting early.')
        quit()

    cf = CocoFilter()

    # Configures outpath paths
    config_output_dir = cf.parse_output_path(args.config_input_dir, args.config_output_dir, args.categories)
    dataset_output_dir = cf.parse_output_path(args.dataset_input_dir, args.dataset_output_dir, args.categories)

    # Iterates through different dataset category for train, val, and test
    for dataset in os.listdir(args.dataset_input_dir):
        # Filters coco.json for only annotations for the categories selected
        cf.filter_json(args.dataset_input_dir + '/' + dataset + '/coco.json',
                    config_output_dir + '/' + dataset + '_coco.json',
                    args.categories)
        # Filters images for the categories selected
        cf.image_sorter(config_output_dir + '/' + dataset + '_coco.json',
                        args.dataset_input_dir,
                        dataset_output_dir)
    
    # Updates .yaml with new train, val, and test paths along with categories
    for file in os.listdir(args.config_input_dir):
        # updates .yaml file with select categories
        if file.endswith('.yaml'):
            cf.filter_yaml(args.config_input_dir + '/' + file,
                      config_output_dir + '/' + file,
                      dataset_output_dir,
                      args.categories)
