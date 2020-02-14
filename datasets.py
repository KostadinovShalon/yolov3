import glob
import json
import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from yolov3.utils.geometry import *
from yolov3.utils.networks import horizontal_flip

import random

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


# noinspection PyTypeChecker
def _get_annotations(json_file):
    """
    Gets the annotations from a VGG Json File
    :param json_file: vgg json file
    :return: a list with the annotation regions from the json file
    """
    with open(json_file, 'r') as f:
        raw_annotations = json.load(f)
    annotations = [dict(filename=ann['filename'], regions=ann['regions']) for _, ann in raw_annotations.items()]
    return annotations


# class ListDataset(Dataset):
#     """
#     @deprecated
#     """
#     def __init__(self, root: str, img_size=416, augment=True, multiscale=True, normalized_labels=True, split='train'):
#         """
#         Dataset for files with the structure
#
#         - class1
#         |  - train
#            |  - image1.jpg
#            |  - image2.jpg
#            |  - ...
#         |  - val
#            |  - image1.jpg
#            |  - image2.jpg
#            |  - ...
#         - class2
#            |  - train...
#
#         :param root: root directory path where data is present
#         :param img_size: rescaled image size
#         :param augment: boolean indicating if data augmentation is going to be applied
#         :param multiscale: indicates if multi-scale must be used
#         :param normalized_labels: indicates if labels in the annotation file are normalized
#         :param split: choose between train and val
#         """
#         self.root = root
#         self.classes = self._find_classes()
#         self.imgs = []
#         for i, cls in enumerate(self.classes):
#             class_path = f"{root}/{cls}/{split}"
#
#             annotation_files = [f for f in os.listdir(class_path) if f.endswith('.json')]
#             img_paths = [f for f in os.listdir(class_path) if is_file(os.path.join(class_path, f)) and is_image_file(f)]
#             annotations = []
#             for annotation_file in annotation_files:
#                 annotations += _get_annotations(os.path.join(class_path, annotation_file))
#             self.imgs += [dict(filename=os.path.join(class_path, img_path),
#                                cls=i,
#                                regions=[ann['regions'] for ann in annotations if ann['filename'] == img_path])
#                           for img_path in img_paths]
#
#         self.img_size = img_size
#         self.augment = augment
#         self.multiscale = multiscale
#         self.normalized_labels = normalized_labels
#         self.min_size = self.img_size - 3 * 32
#         self.max_size = self.img_size + 3 * 32
#         self.batch_count = 0
#
#     def _find_classes(self):
#         """
#         Finds the class folders in a dataset.
#
#         :returns: tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
#
#         :ensures: No class is a subdirectory of another.
#         """
#         if sys.version_info >= (3, 5):
#             # Faster and available in Python 3.5 and above
#             classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
#         else:
#             classes = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
#         classes.sort()
#         return classes
#
#     def regions_to_bounding_boxes(self, regions, img):
#         """
#         Converts a region dict to bounding boxes
#         :param regions: regions dict
#         :param img: original image
#         :return: bounding boxes
#         """
#         _, h, w = img.shape
#
#         # Pad to square resolution
#         img, pad = pad_to_square(img, 0)
#         _, padded_h, padded_w = img.shape
#
#         boxes = None
#         if len(regions) > 0:
#             number_of_annotations = sum(len(individual_region_file) for individual_region_file in regions)
#             boxes = torch.zeros((number_of_annotations, 6))
#             c = 0
#             for file_regions in regions:
#                 for _, region in file_regions.items():
#                     x_points = region['shape_attributes']['all_points_x']
#                     y_points = region['shape_attributes']['all_points_y']
#
#                     h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
#
#                     # Unpadded and unscaled image
#                     x1, x2, y1, y2 = max(x_points), min(x_points), max(y_points), min(y_points)
#                     x1 *= w_factor
#                     x2 *= w_factor
#                     y1 *= h_factor
#                     y2 *= h_factor
#                     # Adding paddding
#                     x1 += pad[0]
#                     y1 += pad[2]
#                     x2 += pad[1]
#                     y2 += pad[2]
#
#                     # Obtaining x, y, w, h
#                     boxes[c, 2] = (x1 + x2) / 2 / padded_w
#                     boxes[c, 3] = (y1 + y2) / 2 / padded_h
#                     boxes[c, 4] = (x1 - x2) * w_factor / padded_w
#                     boxes[c, 5] = (y1 - y2) * h_factor / padded_h
#                     c += 1
#         return img, boxes
#
#     def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
#
#         img = self.imgs[index % len(self.imgs)]
#         img_path = img['filename']
#         img_regions = img['regions']
#         img_cls = img['cls']
#         img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
#         if len(img.shape) != 3:
#             img = img.unsqueeze(0)
#             img = img.expand((3, img.shape[1:]))
#
#         img, targets = self.regions_to_bounding_boxes(img_regions, img)
#         if self.augment:
#             if np.random.random() < 0.5:
#                 img, targets = horizontal_flip(img, targets)
#         targets[:, 1] = img_cls
#         return img_path, img, targets
#
#     def collate_fn(self, batch) -> Tuple[Tuple[Tuple[str]], torch.Tensor, torch.Tensor]:
#         paths, imgs, targets = list(zip(*batch))
#         targets = [boxes for boxes in targets if boxes is not None]
#
#         for i, boxes in enumerate(targets):
#             boxes[:, 0] = i
#         targets = torch.cat(targets, 0)
#         if self.multiscale and self.batch_count % 10 == 0:
#             self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
#
#         imgs = torch.stack([resize(img, self.img_size) for img in imgs])
#         self.batch_count += 1
#         return paths, imgs, targets
#
#     def __len__(self):
#         return len(self.imgs)


class COCODataset(Dataset):

    def __init__(self, root: str,
                 annotations_file: str,
                 img_size=416,
                 augment=True,
                 multiscale=True,
                 normalized_labels=True,
                 partition=None,
                 val_split=0.2,
                 seed=None):
        """
        Dataset for files with the structure

        - root
           |  - image1.jpg
           |  - image2.jpg
           |  - ...

        :param root: root directory path where images are
        :param img_size: rescaled image size
        :param augment: boolean indicating if data augmentation is going to be applied
        :param multiscale: indicates if multi-scale must be used
        :param normalized_labels: indicates if labels in the annotation file are normalized
        :param partition: choose between train and val, None if there training and validation directories are separated.
            If None, then it means validation and training are in different folders and all data is taken
        :param val_split: split of the data for validation
        :param seed: random seed
        """
        self.root = root
        self.anns_file = annotations_file

        with open(self.anns_file, 'r') as f:
            coco_file = json.load(f)

        self.classes = {int(category['id']): category['name'] for category in coco_file['categories']}
        self._c = [i for i, _ in self.classes.items()]
        self._c.sort()
        self._actual_indices = {k: i for i, k in enumerate(self._c)}
        self.imgs = [{"id": int(img['id']),
                      "file_name": img['file_name']} for img in coco_file['images']]
        self.anns = [{"image_id": ann['image_id'],
                      "category_id": ann['category_id'],
                      "bbox": ann['bbox']} for ann in coco_file['annotations']]

        if partition is not None:
            if seed is not None:
                random.seed(seed)
            random.shuffle(self.imgs)

            total_imgs = len(self.imgs)
            total_partition_val = int(total_imgs * val_split)

            self.imgs = self.imgs[:total_partition_val] if partition == "val" else self.imgs[total_partition_val:]

        self.img_size = img_size
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def get_cat_by_positional_id(self, positional_id):
        cat_id = self._c[positional_id]
        return self.classes[cat_id]

    def anns_to_bounding_boxes(self, anns, img):
        """
        Converts a region dict to bounding boxes
        :param anns: lists of anns dictionaries
        :param img: original image
        :return: bounding boxes
        """
        _, h, w = img.shape

        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        boxes = None
        if len(anns) > 0:
            number_of_annotations = len(anns)
            boxes = torch.zeros((number_of_annotations, 6))

            for c, ann in enumerate(anns):
                bbox = ann["bbox"]

                # Unpadded and unscaled image
                # IMPORTANT! THIS IS ONLY FOR COCO DATASET
                x1 = bbox[0] * w_factor
                x2 = (bbox[0] + bbox[2]) * w_factor
                y1 = bbox[1] * h_factor
                y2 = (bbox[1] + bbox[3]) * h_factor
                # Adding paddding
                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[2]

                # Obtaining x, y, w, h
                boxes[c, 1] = ann["category_id"]
                boxes[c, 2] = (x1 + x2) / 2 / padded_w
                boxes[c, 3] = (y1 + y2) / 2 / padded_h
                boxes[c, 4] = (x2 - x1) * w_factor / padded_w
                boxes[c, 5] = (y2 - y1) * h_factor / padded_h
        boxes[:, 1] = torch.tensor(list(map(self._actual_indices.get, boxes[:, 1].tolist())),
                                   device=boxes.device)
        return img, boxes

    def __getitem__(self, index: int) -> Tuple[str, int, torch.Tensor, torch.Tensor]:

        img = self.imgs[index]
        anns = [a for a in self.anns if a["image_id"] == img["id"]]
        img_path = os.path.join(self.root, img["file_name"])
        img_id = img["id"]

        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        img, targets = self.anns_to_bounding_boxes(anns, img)
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horizontal_flip(img, targets)
        return img_path, img_id, img, targets

    def collate_fn(self, batch) -> Tuple[Tuple[str], Tuple[int], torch.Tensor, torch.Tensor]:
        paths, img_ids, imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]

        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, img_ids, imgs, targets

    def __len__(self):
        return len(self.imgs)


class ImageFolder(Dataset):
    """
    Simple Dataset obtaining all files in a folder.
    """

    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.files = [f for f in self.files if is_file(f) and is_image_file(f)]
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        elif img.shape[0] == 1:
            img = img.squeeze()
            img = img.expand((3, img.shape[1:]))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)
