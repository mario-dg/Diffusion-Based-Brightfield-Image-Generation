import os
import cv2
import wandb
import torch
import numpy as np

from tqdm import tqdm
from torch import tensor
from pprint import pprint
from ultralytics import YOLO, RTDETR
from torchmetrics.detection import MeanAveragePrecision


VERSION = "v3"
BASE_ARTIFACTS_PATH = "m-dagraca/Thesis-Research-Detection/"
DOWNLOAD_ARTIFACTS_PATH = "trained_models"
CONFIDENCE = 0.2
IOU = 0.3

TRAINED_MODELS = {
    "real_yolov8s": [YOLO, "run_65yqro5s_model:v0"],
    "10_yolov8s": [YOLO, "run_nek95pk8_model:v0"],
    "30_yolov8s": [YOLO, "run_enq3l6ag_model:v0"],
    "50_yolov8s": [YOLO, "run_jm2ka5vt_model:v0"],
    "real_yolov8m": [YOLO, "run_2u03qam8_model:v0"],
    "10_yolov8m": [YOLO, "run_u0k5i0o0_model:v0"],
    "30_yolov8m": [YOLO, "run_2vwmd3rd_model:v0"],
    "50_yolov8m": [YOLO, "run_5p9pajga_model:v0"],
    "real_yolov8x": [YOLO, "run_v5yjh82t_model:v0"],
    "10_yolov8x": [YOLO, "run_8x1ewnxs_model:v0"],
    "30_yolov8x": [YOLO, "run_mkvuyx4d_model:v0"],
    "50_yolov8x": [YOLO, "run_31313h00_model:v0"],
    "real_yolov9c": [YOLO, "run_k24p50s7_model:v0"],
    "10_yolov9c": [YOLO, "run_1e5fkg4y_model:v0"],
    "30_yolov9c": [YOLO, "run_ft6191yc_model:v0"],
    "50_yolov9c": [YOLO, "run_gnbwoavy_model:v0"],
    "real_yolov9e": [YOLO, "run_5q1n0szd_model:v0"],
    "10_yolov9e": [YOLO, "run_k73ju4gq_model:v0"],
    "30_yolov9e": [YOLO, "run_hdupemzm_model:v0"],
    "50_yolov9e": [YOLO, "run_4j26tact_model:v0"],
    "real_rtdetr-l": [RTDETR, "run_n0r2hzwl_model:v0"],
    "10_rtdetr-l": [RTDETR, "run_puilqzou_model:v0"],
    "30_rtdetr-l": [RTDETR, "run_l5xixxwz_model:v0"],
    "50_rtdetr-l": [RTDETR, "run_vm5rjbvt_model:v0"],
    "real_rtdetr-x": [RTDETR, "run_0evwdjz7_model:v0"],
    "10_rtdetr-x": [RTDETR, "run_asc52d37_model:v0"],
    "30_rtdetr-x": [RTDETR, "run_kp7st0u9_model:v0"],
    "50_rtdetr-x": [RTDETR, "run_8u0jyiql_model:v0"],
}

DATASETS = {
    "scc_cell_detection_real": "datasets/scc_cell_detection_real/",
    "scc_cell_detection_10": "datasets/scc_cell_detection_10/",
    "scc_cell_detection_30": "datasets/scc_cell_detection_30/",
    "scc_cell_detection_50": "datasets/scc_cell_detection_50/",
}


############################################################################
############################# Manual Detection #############################
############################################################################

def read_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        labels.append((class_id, x_center, y_center, width, height))
    return labels


def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    kernel = np.ones((7, 7), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def manual_pred(image_paths, label_paths):
    mAP = MeanAveragePrecision(iou_type="bbox", box_format="xywh", extended_summary=True, max_detection_thresholds=[1, 10, 100])
    for img_path, lbl_path in tqdm(zip(image_paths, label_paths)):
        labels = read_labels(lbl_path)
        contours = process_image(img_path)
        
        height, width = cv2.imread(img_path).shape[:2]
        pred_boxes = [cv2.boundingRect(contour) for contour in contours]
        filtered_boxes = []
        for (x, y, w, h) in pred_boxes:
            if w <= 25 and h >= 25:  # on left or right image edge, possibly
                if h <= 125:
                    filtered_boxes.append((x, y, w, h))
            elif h <= 25 and w >= 25:  # on top or bottom image edge, possibly
                if w <= 125:
                    filtered_boxes.append((x, y, w, h))
            elif (25 <= w <= 125) and (25 <= h <= 125):  # not on any edge, posibly
                filtered_boxes.append((x, y, w, h))

        # pred_boxes = [(x, y, w, h) for x, y, w, h in pred_boxes if (25 <= w <= 100) and (25 <= h <= 100)]
        norm_boxes = [(x / width, y / height, w / width, h / height) for x, y, w, h in filtered_boxes]
        centered_boxes = [(x + w / 2.0, y + h / 2.0, w, h) for x, y, w, h in norm_boxes]
        boxes = []
        ids = []
        for lin in labels:
            ids.append(int(lin[0]))
            boxes.append([float(lin[1]), float(lin[2]), float(lin[3]), float(lin[4])])

        targets = [
            dict(
                boxes = tensor(boxes),
                labels = tensor(ids),
            )
        ]
        preds = [
            dict(
                boxes = tensor(centered_boxes),
                scores = tensor([1.0] * len(centered_boxes)),
                labels = tensor([0] * len(centered_boxes)),
            )
        ]
        mAP.update(preds, targets)

    metrics =  mAP.compute()
    precision_iou_thresholds = metrics['precision'][:, :, 0, 0, -1]
    ap_per_iou_threshold = precision_iou_thresholds.mean(dim=1)
    map50_95 = ap_per_iou_threshold.mean()
    metrics = {k: v for k, v in metrics.items() if isinstance(v, torch.Tensor) and v.shape == torch.Size([]) and v.item() >= 0}
    metrics["map_50-95"] = map50_95
    if 'map_small' in metrics.keys():
        del metrics["map_small"]
    if 'mar_small' in metrics.keys():
        del metrics["mar_small"]
    del metrics["classes"]
    return metrics

############################################################################
############################### AI Detection ###############################
############################################################################

def validate_model(model, model_name, run):
    dataset_dir = DATASETS[f"scc_cell_detection_{model_name.split('_')[0]}"]
    test_images = os.path.join(dataset_dir, "test", "images")
    test_labels = os.path.join(dataset_dir, "test", "labels")

    mAP = MeanAveragePrecision(iou_type="bbox", box_format="xywh", extended_summary=True, max_detection_thresholds=[1, 2, 5])

    for img, lbl in tqdm(zip(os.listdir(test_images), os.listdir(test_labels), strict=True)):
        img = os.path.join(test_images, img)
        lbl = os.path.join(test_labels, lbl)
        results = model(img, imgsz=run.config.imgsz, conf=run.config.conf, iou=run.config.iou, device=run.config.device, save=True, save_txt=True, save_conf=True)

        with open(lbl) as f:
            labels_txt = f.read().split('\n')
            labels_txt = filter(None, map(str.strip, labels_txt))
        
        boxes = []
        ids = []
        for lin in labels_txt:
            id, x, y, w, h = lin.split(' ')
            ids.append(int(id))
            boxes.append([float(x), float(y), float(w), float(h)])

        targets = [
            dict(
                boxes = tensor(boxes),
                labels = tensor(ids),
            )
        ]
        preds = [
            dict(
                boxes = results[0].boxes.xywhn.to("cpu"),
                scores = results[0].boxes.conf.to("cpu"),
                labels = results[0].boxes.cls.to("cpu").int(),
            )
        ]
        mAP.update(preds, targets)

    metrics =  mAP.compute()

    precision_iou_thresholds = metrics['precision'][:, :, 0, 0, -1]
    ap_per_iou_threshold = precision_iou_thresholds.mean(dim=1)
    map50_95 = ap_per_iou_threshold.mean()

    metrics = {k: v for k, v in metrics.items() if isinstance(v, torch.Tensor) and v.shape == torch.Size([]) and v.item() >= 0}
    metrics["map_50-95"] = map50_95

    if 'map_small' in metrics.keys():
        del metrics["map_small"]
    if 'mar_small' in metrics.keys():
        del metrics["mar_small"]
    del metrics["classes"]
    
    return metrics

if __name__ == "__main__":
    for model_name, model_info in TRAINED_MODELS.items():
        model_class, artifact_name = model_info[0], model_info[1]

        run = wandb.init(
            project="Thesis-Research-Detection", 
            name=model_name,
            group=f"{VERSION}_scc_cell_detection_{model_name.split('_')[0]}_test_conf_{CONFIDENCE}", 
            save_code=True,
            tags=['test'],
            config={
                "model": model_name,
                "dataset": f"scc_cell_detection_{model_name.split('_')[0]}",
                "imgsz": 512,
                "batch": 32,
                "conf": CONFIDENCE,
                "device": "cuda:0",
                "iou": IOU,
            })

        model_file = os.path.join(DOWNLOAD_ARTIFACTS_PATH, f"{model_name}_best.pt")
        artifact = run.use_artifact(f"{BASE_ARTIFACTS_PATH}{artifact_name}", type='model')
        if not os.path.exists(model_file):
            artifact_dir = artifact.download(root=DOWNLOAD_ARTIFACTS_PATH)
            os.rename(f"{artifact_dir}/best.pt", model_file)
            
        model = model_class(model_file)

        metrics = validate_model(model, model_name, run)
        pprint(metrics)
        run.log(metrics)
        run.finish()

    #####################################################################################
    # Initially wanted to also compare classical image processing to ai based detection #
    # but decided to focus on the latter, since this is a topic for its own             #
    # bachelor thesis                                                                   #
    #####################################################################################

    # for ds_name, ds_path in DATASETS.items():
    #     run = wandb.init(
    #         project="Thesis-Research-Detection", 
    #         name=f"manual_{ds_name}",
    #         group=f"v1_{ds_name}_manual", 
    #         save_code=True,
    #         tags=['test'],
    #         config={
    #             "dataset": ds_name,
    #             "imgsz": 512,
    #             "min_dim": 25,
    #             "max_dim": 125,
    #             "conf": 0.2,
    #         })
        
    #     image_paths = sorted(glob.glob(f"{ds_path}test/images/*.png"))
    #     label_paths = sorted(glob.glob(f"{ds_path}test/labels/*.txt"))

    #     metrics = manual_pred(image_paths, label_paths)
    #     pprint(metrics)
    #     run.log(metrics)
    #     run.finish()
    #     break  # only need to manually test the split once, since it is the same for every dataset

        
