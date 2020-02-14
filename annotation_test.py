import os
import json
from os import path

import cv2
import numpy as np


def get_annotations(json_file):
    """
    Gets the annotations from a VGG Json File
    :param json_file: vgg json file
    :return: a list with the annotation regions from the json file
    """
    with open(json_file, 'r') as f:
        raw_annotations = json.load(f)
    annotations = [dict(filename=ann['filename'], regions=ann['regions']) for _, ann in raw_annotations.items()]
    return annotations


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    :param filename (string): path to a file
    :param extensions (tuple of strings): extensions to consider (lowercase)

    :returns: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    :param filename: path to a file

    :returns: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def is_file(filename):
    """Checks if a file is an allowed image extension.

    :param filename: path to a file

    :returns: True if the filename ends with a known image extension
    """
    return os.path.isfile(filename)


root_dir = "/home/brian/Documents/Projects/leonardo-object-detection/detection_modules/data/sample/firearm/train"
out_dir = "/home/brian/Documents/Projects/leonardo-object-detection/detection_modules/fornow"

jsonfile = os.path.join(root_dir, "firearm_train.json")
anns = get_annotations(jsonfile)

img_paths = [f for f in os.listdir(root_dir) if is_file(os.path.join(root_dir, f)) and is_image_file(f)]
imgs = [dict(filename=os.path.join(root_dir, img_path), regions=[ann['regions']
                                                                 for ann in anns if ann['filename'] == img_path])
        for img_path in img_paths]

for img_dict in imgs:

    # Create plot
    image = cv2.imread(img_dict["filename"], cv2.IMREAD_COLOR)
    img = image.copy()
    h, w, _ = img.shape
    regions = img_dict["regions"]

    for region_file in regions:
        for _, region in region_file.items():
            x_points = region['shape_attributes']['all_points_x']
            y_points = region['shape_attributes']['all_points_y']
            x1, x2, y1, y2 = max(x_points), min(x_points), max(y_points), min(y_points)

            vertices = list(zip(x_points, y_points))
            vertices = np.array(vertices, np.int32)
            vertices = vertices.reshape((-1, 1, 2))

            cv2.polylines(img, [vertices], True, (0, 255, 255))
            cv2.rectangle(img, (x2, y2), (x1, y1), (255, 0, 0), 3)
    filename = img_dict["filename"].split("/")[-1].split(".")[0]
    cv2.imwrite(os.path.join(out_dir, f"{filename}.png"), img)
