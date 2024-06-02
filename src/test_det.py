import os
import wandb

from ultralytics import YOLO, RTDETR
from ultralytics.utils.callbacks.wb import callbacks as wandb_callbacks


VERSION = "v1"
BASE_ARTIFACTS_PATH = "m-dagraca/Thesis-Research-Detection/"

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

for model_name, model_info in TRAINED_MODELS.items():
    model_class, artifact_name = model_info[0], model_info[1]

    run = wandb.init(
        project="Thesis-Research-Detection", 
        name=model_name,
        group=f"{VERSION}_scc_cell_detection_{model_name.split('_')[0]}_test", 
        save_code=True,
        config={
            "model": model_name,
            "dataset": f"scc_cell_detection_{model_name.split('_')[0]}",
            "imgsz": 512,
            "batch": 32,
            "conf": 0.5,
            "device": "cuda:0",
        })

    artifact = run.use_artifact(f"{BASE_ARTIFACTS_PATH}{artifact_name}", type='model', aliases=f"{model_name}_best.pt")
    artifact_dir = artifact.download(root="trained_models")
    model_file = os.path.join(artifact_dir, f"{model_name}_best.pt")
    os.rename(f"{artifact_dir}/best.pt", model_file)

    model = YOLO(model_file)
    for cb_event, cb in wandb_callbacks.items():
        model.add_callback(cb_event, cb)

    metrics = model.val(
        imgsz=run.config.imgsz,
        batch=run.config.batch,
        conf=run.config.conf,
        save_json=True,
        save_hybrid=True,
        device=run.config.device,
        plots=True,
        split="test",
    )
    print()
    print()
    print()
    print()
    print()
    print()
    print(metrics)
    wandb.finish()
    break

# run = wandb.init(
#                 project="Thesis-Research-Detection", 
#                 name=model_name,
#                 group=f"{VERSION}_{ds_name}", 
#                 save_code=True, )
# artifact = run.use_artifact('m-dagraca/Thesis-Research-Detection/run_u0k5i0o0_model:v0', type='model')
# artifact_dir = artifact.download()