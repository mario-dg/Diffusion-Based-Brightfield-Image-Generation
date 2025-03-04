import os
import shutil
import datasets
import supervision as sv

from tqdm import tqdm

generated_ds = sv.DetectionDataset.from_yolo("datasets/scc_cell_detection_generated/train/images", "datasets/scc_cell_detection_generated/train/labels", "datasets/scc_cell_detection_generated/data.yaml")
print(f"Generated dataset: {len(generated_ds)} images")

real_ds = datasets.load_dataset("mario-dg/brightfield-microscopy-scc-filtered", cache_dir=".cache/")

# convert json labels from huggingface to yolo format
splits = real_ds.keys()
shutil.rmtree("datasets/scc_real", ignore_errors=True)
for split in splits:
    ds_dir = f"datasets/scc_real/{split}"
    images_dir = f"{ds_dir}/images"
    labels_dir = f"{ds_dir}/labels"
    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    for index, item in tqdm(enumerate(real_ds[split]), desc=split, total=len(real_ds[split])):
        image_file = f"{ds_dir}/images/real_cell_image_{index:04d}.png"
        item["image"].save(image_file)
        with open(f"{ds_dir}/labels/real_cell_image_{index:04d}.txt", 'w') as f:
            for det_index, det in enumerate(item['objects']['bbox']):
                f.write(f"{item['objects']['categories'][det_index]} {det[0]} {det[1]} {det[2]} {det[3]}\n")

real_train_ds = sv.DetectionDataset.from_yolo("datasets/scc_real/train/images", "datasets/scc_real/train/labels", "datasets/scc_real/data.yaml")
real_val_ds = sv.DetectionDataset.from_yolo("datasets/scc_real/validation/images", "datasets/scc_real/validation/labels", "datasets/scc_real/data.yaml")
real_test_ds = sv.DetectionDataset.from_yolo("datasets/scc_real/test/images", "datasets/scc_real/test/labels", "datasets/scc_real/data.yaml")
print(f"Real dataset: {len(real_train_ds)} train images")
print(f"Real dataset: {len(real_val_ds)} validation images")
print(f"Real dataset: {len(real_test_ds)} test images")

NUM_TRAIN_IMAGES = 5000
REAL_10 = int(NUM_TRAIN_IMAGES * 0.9)
REAL_30 = int(NUM_TRAIN_IMAGES * 0.7)
REAL_50 = int(NUM_TRAIN_IMAGES * 0.5)
SCC_10 = NUM_TRAIN_IMAGES - REAL_10
SCC_30 = NUM_TRAIN_IMAGES - REAL_30
SCC_50 = NUM_TRAIN_IMAGES - REAL_50
SCC_ADD_10 = NUM_TRAIN_IMAGES * 0.1
SCC_ADD_30 = NUM_TRAIN_IMAGES * 0.3
SCC_ADD_50 = NUM_TRAIN_IMAGES * 0.5


scc_10 = sv.DetectionDataset.merge([sv.DetectionDataset(
                                                    classes=generated_ds.classes,
                                                    images={name: generated_ds.images[name] for name in list(generated_ds.images.keys())[:SCC_10]},
                                                    annotations={name: generated_ds.annotations[name] for name in list(generated_ds.images.keys())[:SCC_10]},
                                                ), 
                                                    sv.DetectionDataset(
                                                    classes=real_train_ds.classes,
                                                    images={name: real_train_ds.images[name] for name in list(real_train_ds.images.keys())[:REAL_10]},
                                                    annotations={name: real_train_ds.annotations[name] for name in list(real_train_ds.images.keys())[:REAL_10]},
                                                )])
scc_30 = sv.DetectionDataset.merge([sv.DetectionDataset(
                                                    classes=generated_ds.classes,
                                                    images={name: generated_ds.images[name] for name in list(generated_ds.images.keys())[:SCC_30]},
                                                    annotations={name: generated_ds.annotations[name] for name in list(generated_ds.images.keys())[:SCC_30]},
                                                    ), 
                                                    sv.DetectionDataset(
                                                    classes=real_train_ds.classes,
                                                    images={name: real_train_ds.images[name] for name in list(real_train_ds.images.keys())[:REAL_30]},
                                                    annotations={name: real_train_ds.annotations[name] for name in list(real_train_ds.images.keys())[:REAL_30]},
                                                )])
scc_50 = sv.DetectionDataset.merge([sv.DetectionDataset(
                                                    classes=generated_ds.classes,
                                                    images={name: generated_ds.images[name] for name in list(generated_ds.images.keys())[:SCC_50]},
                                                    annotations={name: generated_ds.annotations[name] for name in list(generated_ds.images.keys())[:SCC_50]},
                                                    ), 
                                                    sv.DetectionDataset(
                                                    classes=real_train_ds.classes,
                                                    images={name: real_train_ds.images[name] for name in list(real_train_ds.images.keys())[:REAL_50]},
                                                    annotations={name: real_train_ds.annotations[name] for name in list(real_train_ds.images.keys())[:REAL_50]},
                                                )])

scc_add_10 = sv.DetectionDataset.merge([sv.DetectionDataset(
                                                    classes=generated_ds.classes,
                                                    images={name: generated_ds.images[name] for name in list(generated_ds.images.keys())[:SCC_ADD_10]},
                                                    annotations={name: generated_ds.annotations[name] for name in list(generated_ds.images.keys())[:SCC_ADD_10]},
                                                ), 
                                                    sv.DetectionDataset(
                                                    classes=real_train_ds.classes,
                                                    images={name: real_train_ds.images[name] for name in list(real_train_ds.images.keys())[:NUM_TRAIN_IMAGES]},
                                                    annotations={name: real_train_ds.annotations[name] for name in list(real_train_ds.images.keys())[:NUM_TRAIN_IMAGES]},
                                                )])
scc_add_30 = sv.DetectionDataset.merge([sv.DetectionDataset(
                                                    classes=generated_ds.classes,
                                                    images={name: generated_ds.images[name] for name in list(generated_ds.images.keys())[:SCC_ADD_30]},
                                                    annotations={name: generated_ds.annotations[name] for name in list(generated_ds.images.keys())[:SCC_ADD_30]},
                                                ), 
                                                    sv.DetectionDataset(
                                                    classes=real_train_ds.classes,
                                                    images={name: real_train_ds.images[name] for name in list(real_train_ds.images.keys())[:NUM_TRAIN_IMAGES]},
                                                    annotations={name: real_train_ds.annotations[name] for name in list(real_train_ds.images.keys())[:NUM_TRAIN_IMAGES]},
                                                )])
scc_add_50 = sv.DetectionDataset.merge([sv.DetectionDataset(
                                                    classes=generated_ds.classes,
                                                    images={name: generated_ds.images[name] for name in list(generated_ds.images.keys())[:SCC_ADD_50]},
                                                    annotations={name: generated_ds.annotations[name] for name in list(generated_ds.images.keys())[:SCC_ADD_50]},
                                                ), 
                                                    sv.DetectionDataset(
                                                    classes=real_train_ds.classes,
                                                    images={name: real_train_ds.images[name] for name in list(real_train_ds.images.keys())[:NUM_TRAIN_IMAGES]},
                                                    annotations={name: real_train_ds.annotations[name] for name in list(real_train_ds.images.keys())[:NUM_TRAIN_IMAGES]},
                                                )])

scc_10.as_yolo("datasets/scc_10/train/images", "datasets/scc_10/train/labels", "datasets/scc_10/scc_10.yaml")
scc_30.as_yolo("datasets/scc_30/train/images", "datasets/scc_30/train/labels", "datasets/scc_30/scc_30.yaml")
scc_50.as_yolo("datasets/scc_50/train/images", "datasets/scc_50/train/labels", "datasets/scc_50/scc_50.yaml")
scc_add_10.as_yolo("datasets/scc_add_10/train/images", "datasets/scc_add_10/train/labels", "datasets/scc_add_10/scc_add_10.yaml")
scc_add_30.as_yolo("datasets/scc_add_30/train/images", "datasets/scc_add_30/train/labels", "datasets/scc_add_30/scc_add_30.yaml")
scc_add_50.as_yolo("datasets/scc_add_50/train/images", "datasets/scc_add_50/train/labels", "datasets/scc_add_50/scc_add_50.yaml")