import wandb

from ultralytics import YOLO, RTDETR
from ultralytics.utils.callbacks.wb import callbacks as wandb_callbacks

VERSION = "v1"
BASE_MODELS_PATH = "models/base/"

MODELS = {
    "yolov8s": [YOLO, f"{BASE_MODELS_PATH}yolov8s.pt"],
    "yolov8m": [YOLO, f"{BASE_MODELS_PATH}yolov8m.pt"],
    "yolov8x": [YOLO, f"{BASE_MODELS_PATH}yolov8x.pt"],
    "yolov9c": [YOLO, f"{BASE_MODELS_PATH}yolov9c.pt"],
    "yolov9e": [YOLO, f"{BASE_MODELS_PATH}yolov9e.pt"],
    "rtdetr-l": [RTDETR, f"{BASE_MODELS_PATH}rtdetr-l.pt"],
    "rtdetr-x": [RTDETR, f"{BASE_MODELS_PATH}rtdetr-x.pt"],
}

DATASETS = {
    "scc_cell_detection_real": "datasets/scc_cell_detection_real/data.yaml",
    "scc_cell_detection_10": "datasets/scc_cell_detection_10/data.yaml",
    "scc_cell_detection_30": "datasets/scc_cell_detection_30/data.yaml",
    "scc_cell_detection_50": "datasets/scc_cell_detection_50/data.yaml",
}

EPOCHS = 200
PATIENCE = 35
BATCH = 16
IMGSZ = 512
DEVICE = "cuda:0"
SAVE_PERIOD = 10

if __name__ == "__main__":
    for ds_name, ds_path in DATASETS.items():
        for model_name, model_info in MODELS.items():
            run = wandb.init(
                project="Thesis-Research-Detection", 
                name=model_name,
                group=f"{VERSION}_{ds_name}", 
                save_code=True, 
                config={
                    "model": model_name,
                    "dataset": ds_name,
                    "epochs": EPOCHS,
                    "patience": PATIENCE,
                    "batch": BATCH,
                    "imgsz": IMGSZ,
                    "device": DEVICE,
                })

            model_class, model_path = model_info[0], model_info[1]
            model = model_class(model_path)
            for cb_event, cb in wandb_callbacks.items():
                model.add_callback(cb_event, cb)

            model.train(
                data=ds_path,
                epochs=EPOCHS,
                patience=PATIENCE,
                batch=BATCH,
                imgsz=IMGSZ,
                device=DEVICE,
                save_period=SAVE_PERIOD,
                plots=True,
                project=f"{VERSION}_{ds_name}_{model_name}",
            )
