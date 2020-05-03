import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from yolov3.datasets import COCODataset
from yolov3.models import YOLOv3
from yolov3.utils.boxes import xywh2xyxy, non_max_suppression, rescale_boxes
from yolov3.utils.parser import ConfigParser
from yolov3.utils.statistics import get_batch_statistics, ap_per_class

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(dataset, model, iou_thres, conf_thres, nms_thres, img_size,
             workers, weights_path=None, bs=1, return_detections=False):

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False,
        num_workers=workers,
        collate_fn=dataset.collate_fn
    )
    if weights_path is not None:
        if weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path))
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    detections = []
    evaluation_loss = 0
    for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target

        if device:
            imgs, targets = imgs.to(device), targets.to(device)
        imgs.requires_grad = False

        with torch.no_grad():
            loss, outputs = model(imgs, targets)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        evaluation_loss += loss.item()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        # for img_path, img_id, dets in zip(file_names, img_ids, outputs):
        #     if dets is not None:
        #         for det in dets:
        #             w, h = Image.open(img_path).convert('RGB').size
        #             d = det.clone()
        #             d[:4] = rescale_boxes(d[:4].unsqueeze(0), img_size, (h, w)).squeeze()
        #             d = d.tolist()
        #             detections.append({
        #                 "image_id": img_id,
        #                 "category_id": dataset._c[int(d[-1])],
        #                 "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]],
        #                 "score": d[-2]
        #             })

        sample_metrics += get_batch_statistics(outputs, targets.cpu(), iou_threshold=iou_thres)
    evaluation_loss /= len(dataloader)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    p, r, ap, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    if return_detections:
        return evaluation_loss, p, r, ap, f1, ap_class, detections
    else:
        return evaluation_loss, p, r, ap, f1, ap_class


def test(config_file):
    opts = ConfigParser(config_file)

    print("Compute mAP...")

    dataset = COCODataset(opts.test["dir"],
                          annotations_file=opts.test["annotation_file"],
                          augment=False,
                          multiscale=False,
                          normalized_labels=opts.test["normalized"],)

    # Initiate model
    model = YOLOv3(len(dataset.classes), anchors=opts.anchors).to(device)

    (precision, recall, AP, f1, ap_class), detections = evaluate(
        dataset,
        model,
        opts.test["iou_thres"],
        opts.test["conf_thres"],
        opts.test["nms_thres"],
        opts.img_size,
        opts.workers,
        opts.test["weights_file"],
        opts.test["batch_size"],
        return_detections=True
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({ dataset.get_cat_by_positional_id(c)}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")

    json_file_name = opts.test["json_file_output"]

    with open(json_file_name, 'w') as f:
        json.dump(detections, f)
