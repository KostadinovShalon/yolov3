from yolov3.models import YOLOv3
from yolov3.utils.boxes import non_max_suppression, rescale_boxes
from yolov3.utils.parse import ConfigParser
from yolov3.datasets import *

import os
import time
import datetime
import torch
from torch.utils.data import DataLoader
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hsv2bgr(hsv):
    h, s, v = hsv
    c = s * v
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = v - c
    if 0 <= h < 60:
        b, g, r = 0, x, c
    elif 60 <= h < 120:
        b, g, r = 0, c, x
    elif 120 <= h < 180:
        b, g, r = x, c, 0
    elif 180 <= h < 240:
        b, g, r = c, x, 0
    elif 240 <= h < 270:
        b, g, r = c, 0, x
    else:
        b, g, r = x, 0, c
    return (b + m) * 255, (g + m) * 255, (r + m) * 255


def detect(config_file):
    opts = ConfigParser(config_file)
    assert opts.inference["classes"] is not None

    dataset = ImageFolder(opts.inference["dir"], img_size=opts.img_size)

    # Set up
    model = YOLOv3(len(opts.inference["classes"]), anchors=opts.anchors).to(device)

    if opts.inference["weights_file"].endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opts.inference["weights_file"])
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opts.inference["weights_file"]))

    model.eval()  # Set in evaluation mode

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opts.inference["batch_size"], shuffle=False,
        num_workers=4
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")

    n_imgs = opts.inference["max_images"]
    img_counter = 0
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = input_imgs.to(device)

        prev_time = time.time()
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opts.inference["conf_thres"],
                                             opts.inference["nms_thres"])
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        # Log progress
        print("\t+ Batch %d, Inference Time (including NMS): %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
        img_counter += len(img_paths)
        if 0 < n_imgs <= img_counter:
            break

    colors = [hsv2bgr((h if h < 30 else 360 - h, 1, 1)) for h in range(0, 60, 60 // len(opts.inference["classes"]))]
    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        img = image.copy()

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opts.img_size, img.shape[:2])

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (opts.inference["classes"][int(cls_pred)], cls_conf.item()))

                color = colors[int(cls_pred)]

                # Add label
                font = cv2.FONT_HERSHEY_TRIPLEX
                text = opts.inference["classes"][int(cls_pred)]
                font_scale = 1
                thickness = 2

                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                x_centered = x1 - (text_size[0] - abs(x2 - x1)) // 2
                # Create a Rectangle patch
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                cv2.putText(image, opts.inference["classes"][int(cls_pred)], (x_centered, y1), font, 1, color, 2,
                            cv2.LINE_AA)

                alpha = 0.4
                cv2.addWeighted(image, alpha, img, 1 - alpha, 0, img)

        # Save generated image with detections
        filename = path.split("/")[-1].split(".")[0]
        cv2.imwrite(os.path.join(opts.inference_dir, f"{filename}.png"), image)


def draw_detections(image, detection, text, color, box_normalized=True):
    cls, x, y, w, h = detection[-5], detection[-4], detection[-3], detection[-2], detection[-1]
    im_h, im_w = image.shape[:2]

    w_factor, h_factor = (im_w, im_h) if box_normalized else (1, 1)

    x = x * w_factor
    y = y * h_factor
    w = w * w_factor
    h = h * h_factor
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    thickness = 2

    x1 = x - w / 2
    x2 = x + w / 2
    y1 = y - h / 2
    y2 = y + h / 2

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x_centered = x - (text_size[0] - w) // 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    cv2.putText(image, text, (x_centered, y), font, 1, color, 2, cv2.LINE_AA)

    # alpha = 0.4
    # cv2.addWeighted(image, alpha, img, 1 - alpha, 0, img)
