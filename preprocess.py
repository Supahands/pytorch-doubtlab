import argparse
import json
import os
import shutil

import cv2
import pandas as pd
import requests


def get_classes(categories):

    """
    Maps the classes/categories in a dataset to indices.
    Returns a dict of this mapping.
    """

    class_to_idx = {}

    for cat in categories:
        class_name = cat["name"].replace(" ", "_")
        class_id = cat['id']
        class_to_idx[class_name] = class_id

    return class_to_idx

def create_data_structure(class_to_idx):

    """
    Creates the data directory structure that is expected by the
    training/testing scripts. See readme for structure map.
    """

    splits = ["train", "valid", "test"]

    for key,_ in class_to_idx.items():
        for split in splits:
            new_path = f"Dataset_x/{split}/{key}"
            os.makedirs(new_path, exist_ok=True)

    os.mkdir("temp")

def get_category_counts(annos):

    """
    Counts the number of annotations in the dataset under each class.
    Returns a dict of class:count
    """

    counts = {}

    for anno in annos:
        cat_id = anno['category_id']
        if cat_id in counts:
            counts[cat_id] += 1
        else:
            counts[cat_id] = 1

    return counts

def get_split(counts, train=0.7, valid=0.15, test=0.15):

    """
    Splits the no. of annotations in each class into the ratio specified above
    for train, valid, and test sets.
    """

    split_size = {}
    sets = ["train", "valid", "test"]

    for cat,count in counts.items():
        split_size[cat] = {}
        for split_set in sets:
            if split_set == "train":
                split_size[cat]["train"] = int(train*count)
            elif split_set == "valid":
                split_size[cat]["valid"] = int(valid*count)
            elif split_set == "test":
                split_size[cat]["test"] = int(test*count)

    for cat,split in split_size.items():
        total_annos = 0
        for split_set in split:
            total_annos += split_size[cat][split_set]
        diff = counts[cat] - total_annos
        split_size[cat]["train"] += diff

    return split_size

def download_images(csv_path):

    """
    Downloads the images from the csv URLs 
    Stores them in a temporary "temp" folder
    """

    df = pd.read_csv(csv_path)

    for _,row in df.iterrows():

        img_url = row['Image Annotation URL']
        filename = img_url.split('/')[-1]
        img_data = requests.get(img_url).content
        with open(f'temp/{filename}', 'wb') as handler:
            handler.write(img_data)

def crop_annos(split_size, annos, images, idx_to_class):

    """
    Crops annos into their respective folders based on the count of each
    class
    """

    for anno in annos:
        
        image_id = anno['image_id']
        anno_id = anno["id"]
        cat_id = anno['category_id']
        cat_name = idx_to_class[cat_id]
        for img in images:
            if img['id'] == image_id:
                image_name = img['file_name']
        filepath = f"temp/{image_name}"
  
        target_train = split_size[cat_id]["train"]
        target_val = split_size[cat_id]["valid"]

        coords = anno["bbox"]
        y1 = int(coords[1])
        y2 = int(y1 + coords[3])
        x1 = int(coords[0])
        x2 = int(x1 + coords[2])

        img = cv2.imread(filepath)
        crop_img = img[y1:y2, x1:x2]

        curr_train = len(os.listdir(f'Dataset_x/train/{cat_name}'))

        if curr_train < target_train:
            image_path = f"Dataset_x/train/{cat_name}/{anno_id}_{image_name}"
        else:
            curr_val = len(os.listdir(f'Dataset_x/valid/{cat_name}'))
            if curr_val < target_val:
                image_path = f"Dataset_x/valid/{cat_name}/{anno_id}_{image_name}"
            else:
                image_path = f"Dataset_x/test/{cat_name}/{anno_id}_{image_name}"

        try:
            cv2.imwrite(image_path, crop_img)
        except:
            continue

def cleanup():

    """
    Removes the temp dir with the whole/uncropped images
    """

    shutil.rmtree('temp')

if __name__ =="__main__":

    parser = argparse.ArgumentParser(
		description="",
		formatter_class=argparse.RawTextHelpFormatter,
		add_help=False
	)

    parser.add_argument("-j", '--json', type=str, help="path to coco json file")
    parser.add_argument("-c", "--csv", type=str, help="path to csv file")
    args = parser.parse_args()
    json_path = args.json
    csv_path = args.csv

    f = open(json_path)
    coco_data = json.load(f)

    categories = coco_data["categories"]
    annos = coco_data['annotations']
    images = coco_data["images"]

    class_to_idx = get_classes(categories)
    idx_to_class = {value:key for key,value in class_to_idx.items()}

    create_data_structure(class_to_idx)
    category_counts = get_category_counts(annos)
    split_size = get_split(category_counts)
    download_images(csv_path)
    crop_annos(split_size, annos, images, idx_to_class)
    cleanup()



    

    












