from yolov3.models import *
from yolov3.datasets import *
from yolov3.utils.networks import weights_init_normal
from yolov3.test import evaluate

from terminaltables import AsciiTable

import os
import time
import datetime

import torch
from torch.utils.data import DataLoader

from yolov3.utils.parse import ConfigParser
import visdom
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def estimate_reimaining_time(start_time, dataloader, current_batch):
    epoch_batches_left = len(dataloader) - (current_batch + 1)
    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (current_batch + 1))
    return f"\n---- ETA {time_left}"


def plot(train_loss, ap, title, vis, win=None):
    f = plt.figure(figsize=(16, 8))
    f.suptitle(title)
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_loss)
    ax.set_title('Train Loss.')
    ax.set_xlabel('Epoch')

    ax = f.add_subplot(1, 2, 2)
    ax.plot(ap)
    ax.set_title('Test AP.')
    ax.set_xlabel('Epoch')

    if win is None:
        win = vis.matplot(f)
    else:
        vis.matplot(f, win=win)

    plt.close(f)

    return win


def train(config_file):
    opts = ConfigParser(config_file)

    seed = random.randint(1, 100000)
    # Get dataloader
    train_dataset_args = dict(
        root=opts.train["dir"],
        annotations_file=opts.train["annotation_file"],
        augment=False,
        multiscale=opts.train["multiscale_training"],
        normalized_labels=opts.train["normalized"],
    )

    val_dataset_args = dict(
        root=opts.train["dir"],
        annotations_file=opts.train["annotation_file"],
        augment=False,
        multiscale=False,
        normalized_labels=opts.train["normalized"]
    )

    if opts.train["val_dir"] is None:
        train_dataset_args["partition"] = "train"
        train_dataset_args["val_split"] = opts.train["val_split"]
        train_dataset_args["seed"] = seed

        val_dataset_args["partition"] = "val"
        val_dataset_args["val_split"] = opts.train["val_split"]
        val_dataset_args["seed"] = seed
    else:
        val_dataset_args["root"] = opts.train["val_dir"]
        val_dataset_args["annotations_file"] = opts.train["val_annotation_file"]

    dataset = COCODataset(**train_dataset_args)

    # Get eval dataloader
    eval_dataset = COCODataset(**val_dataset_args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.train["batch_size"],
        shuffle=True,
        num_workers=opts.workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    # Initiate model
    model = YOLOv3(len(dataset.classes), anchors=opts.anchors).to(device)
    model.apply(weights_init_normal)

    if opts.train["pretrained_weights"]:
        # noinspection PyTypeChecker
        if opts.train["pretrained_weights"].endswith(".pth"):
            model.load_state_dict(torch.load(opts.train["pretrained_weights"]))
        else:
            model.load_yolov3_weights(opts.train["pretrained_weights"])

    if opts.train["optimizer"]["type"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opts.train["optimizer"]["lr"],
                                     weight_decay=opts.train["optimizer"]["decay"])
    else:
        momentum = opts.train["optimizer"]["momentum"]
        if momentum is None:
            momentum = 0.9
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=opts.train["optimizer"]["lr"],
                                    weight_decay=opts.train["optimizer"]["decay"],
                                    momentum=momentum)
    steps = opts.train["optimizer"]["scheduler_milestones"]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, opts.train["optimizer"]["gamma"]) \
        if steps else None

    output_file = os.path.join(opts.weights_dir, opts.train["output_name"])

    viz = visdom.Visdom(server=opts.visdom["host"], port=opts.visdom["port"]) if opts.visdom["show"] else None

    with open(os.path.join(opts.logs_dir, "stats.txt"), 'w') as f_stat:

        best_ap = []
        best_map = 0
        best_epoch = 0
        win = None
        losses = []
        aps = []

        for epoch in range(opts.train["epochs"]):
            model.train()
            start_time = time.time()
            for batch_i, (file_names, _, imgs, targets) in enumerate(dataloader):
                batches_done = len(dataloader) * epoch + batch_i

                imgs, targets = imgs.to(device), targets.to(device)
                targets.requires_grad = False

                loss, outputs = model(imgs, targets)
                loss.backward()

                if not batches_done % opts.train["gradient_accumulations"]:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()

                log_str = f"\n---- [Epoch {epoch}/{opts.train['epochs']}, Batch {batch_i}/{len(dataloader)}] ----\n"
                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
                # Log metrics at each YOLO layer
                metric_table += model.get_metrics()
                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {float(loss.item())}"

                # Determine approximate time left for epoch
                log_str += estimate_reimaining_time(start_time, dataloader, batch_i)

                print(log_str)

                model.seen += imgs.size(0)

            losses.append(loss.item())
            if scheduler:
                scheduler.step()

            if epoch % opts.train["evaluation_interval"] == 0:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                precision, recall, AP, f1, ap_class = evaluate(
                    eval_dataset,
                    model,
                    iou_thres=opts.train["iou_thres"],
                    conf_thres=opts.train["conf_thres"],
                    nms_thres=opts.train["nms_thres"],
                    img_size=opts.img_size,
                    workers=opts.workers,
                    bs=opts.train["batch_size"]
                )

                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, dataset.get_cat_by_positional_id(c), "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")
                f_stat.write(f'---- epoch {epoch}\n{str(AsciiTable(ap_table).table)}\n---- mAP {str(AP.mean())}\n')

                if AP.mean() >= best_map:
                    best_ap = list("{0:0.4f}".format(i) for i in AP)
                    best_map = AP.mean()
                    best_epoch = epoch
                    torch.save(model.state_dict(), output_file)

                best_map_str = "{0:.4f}".format(best_map)

                print(f'<< Best Results|| Epoch {best_epoch} | Class {best_ap} | mAP {best_map_str} >>')
                f_stat.write(f'Best Results-->> Epoch {best_epoch} | Class {best_ap} | mAP {best_map_str}\n')

                aps.append(AP.mean())
            if viz:
                win = plot(losses, aps, 'YOLOv3', viz, win)

            if epoch % opts.train["checkpoint_interval"] == 0:
                torch.save(model.state_dict(), os.path.join(opts.checkpoints_dir, f"yolov3_ckpt_{epoch}.pth"))
