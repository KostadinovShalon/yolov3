import copy

from tqdm import tqdm

from yolov3.models import YOLOv3
from yolov3.utils.boxes import non_max_suppression, rescale_boxes
from yolov3.datasets import *

import os
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from yolov3.utils.visualization import hsv2bgr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dirs(inference_dir, classes):
    os.makedirs(os.path.join(inference_dir, "none"), exist_ok=True)
    for c in classes:
        os.makedirs(os.path.join(inference_dir, c), exist_ok=True)
        for k in classes:
            if k != c:
                os.makedirs(os.path.join(inference_dir, c, k), exist_ok=True)
        os.makedirs(os.path.join(inference_dir, c, 'none'), exist_ok=True)
        os.makedirs(os.path.join(inference_dir, 'none', c), exist_ok=True)


def detect(parser):
    classes = parser.inference["classes"]
    annotations = None
    if parser.inference["annotation_file"] is not None:
        with open(parser.inference["annotation_file"], 'r') as f:
            coco = json.load(f)
        annotations = coco['annotations']
        categories = sorted(coco['categories'], key=lambda key: key['id'])
        classes = [c['name'] for c in categories]
    save_structured = annotations is not None and parser.inference["save_structured"]
    if save_structured:
        make_dirs(parser.inference_dir, classes)

    # Set up
    model = YOLOv3(len(classes), anchors=parser.anchors).to(device)

    if parser.inference["weights_file"].endswith(".weights"):
        model.load_darknet_weights(parser.inference["weights_file"])
    else:
        model.load_state_dict(torch.load(parser.inference["weights_file"]))

    model.eval()
    with_gt = parser.inference["with_gt"] and annotations is not None
    dataloader_params = {"batch_size": parser.inference["batch_size"], "shuffle": False,
                         "num_workers": parser.workers}
    if with_gt:
        dataset = COCODataset(parser.inference["dir"],
                              annotations_file=parser.inference["annotation_file"],
                              augment=False,
                              multiscale=False,
                              normalized_labels=parser.inference["normalized"],
                              include_filenames=True)
        dataloader_params["collate_fn"] = dataset.collate_fn
    else:
        dataset = ImageFolder(parser.inference["dir"], img_size=parser.img_size)
    dataloader_params["dataset"] = dataset
    dataloader = torch.utils.data.DataLoader(**dataloader_params)

    n_imgs = parser.inference["max_images"]
    img_counter = 0

    for data in tqdm(dataloader, desc="Detecting objects and saving images"):
        # Configure input
        targets = None
        gt = None
        if not with_gt:
            img_paths, imgs = data
        else:
            img_paths, _, imgs, targets = data
            targets = targets.to(device)

        imgs = imgs.to(device)
        # Get detections
        with torch.no_grad():
            detections = model(imgs)
            detections = non_max_suppression(detections, parser.inference["conf_thres"],
                                             parser.inference["nms_thres"])

        def save_no_gt(img_path, filename, img_detection):
            if img_detection is None or len(img_detection) == 0:
                out_path = os.path.join(parser.inference_dir, 'none', f"{filename}.png")
                draw_detections(img_path, out_path, img_detection, classes, parser.img_size)
            else:
                detected_classes = img_detection.t()[-1]
                for detected_class in detected_classes:
                    cls = classes[int(detected_class)]
                    out_path = os.path.join(parser.inference_dir, 'none', cls, f"{filename}.png")
                    draw_detections(img_path, out_path, img_detection, classes, parser.img_size)

        for batch_id, (img_path, img_detection) in enumerate(zip(img_paths, detections)):
            if targets is not None:
                gt = torch.stack([t for t in targets if t[0] == batch_id])
            filename = img_path.split("/")[-1].split(".")[0]
            if save_structured and gt is not None:
                if len(gt) == 0:
                    save_no_gt(img_path, filename, img_detection)
                else:
                    gt_classes = gt.t()[1]
                    for gt_class in gt_classes:
                        if gt_class >= len(classes):
                            save_no_gt(img_path, filename, img_detection)
                        else:
                            g_cls = classes[int(gt_class)]
                            if img_detection is None or len(img_detection) == 0:
                                out_path = os.path.join(parser.inference_dir, g_cls, 'none', f"{filename}.png")
                                draw_detections(img_path, out_path, img_detection, classes, parser.img_size, gt=gt)
                            else:
                                detected_classes = img_detection.t()[-1]
                                for detected_class in detected_classes:
                                    cls = classes[int(detected_class)]
                                    if cls == g_cls:
                                        out_path = os.path.join(parser.inference_dir, cls, f"{filename}.png")
                                    else:
                                        out_path = os.path.join(parser.inference_dir, g_cls, cls, f"{filename}.png")
                                    draw_detections(img_path, out_path, img_detection, classes, parser.img_size, gt=gt)
            else:
                out_path = os.path.join(parser.inference_dir, f"{filename}.png")
                draw_detections(img_path, out_path, img_detection, classes, parser.img_size, gt=gt)

        img_counter += len(img_paths)
        if 0 < n_imgs <= img_counter:
            break


def draw_detections(image_path, out_path, detections, classes, model_img_size, colors=None, gt=None):
    if colors is None:
        colors = [hsv2bgr((h, 1, 1)) for h in range(120, 240, 120 // len(classes))]

    image = Image.open(image_path).convert("RGB")
    drawer = ImageDraw.Draw(image)
    im_w, im_h = image.size

    if gt is not None:
        g = copy.deepcopy(gt)
        g[:, 2:6] = rescale_boxes(g[:, 2:6], model_img_size, (im_h, im_w), normalized=True, xywh=True)
        for _, cls_idx, x1, y1, x2, y2 in g:
            cls = classes[int(cls_idx)]
            w = x2 - x1
            text_size = drawer.textsize(cls)
            x_centered = x1 - (text_size[0] - w) // 2
            drawer.rectangle([x1, y1, x2, y2], outline="black", width=1)
            drawer.text((x_centered, y2), cls, fill="black")

    # Detections adjustment
    if detections is not None:
        rescaled_detections = copy.deepcopy(detections)
        rescaled_detections = rescale_boxes(rescaled_detections, model_img_size, (im_h, im_w))
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in rescaled_detections:
            cls = classes[int(cls_pred)]
            color = colors[int(cls_pred)]
            w = x2 - x1
            text_size = drawer.textsize(cls)
            x_centered = x1 - (text_size[0] - w) // 2
            drawer.rectangle([x1, y1, x2, y2], outline=color, width=2)
            drawer.text((x_centered, y1 - text_size[1]), cls, fill=color)
    image.save(out_path, "PNG")
