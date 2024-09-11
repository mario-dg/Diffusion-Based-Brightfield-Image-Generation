import os
import shutil
import datasets
import supervision as sv

from tqdm import tqdm

fake_ds = sv.DetectionDataset.from_yolo("datasets/scc_cell_detection_fake/train/images", "datasets/scc_cell_detection_fake/train/labels", "datasets/scc_cell_detection_fake/data.yaml")
print(f"Fake dataset: {len(fake_ds)} images")

real_ds = datasets.load_dataset("mario-dg/brightfield-microscopy-scc-filtered", cache_dir=".cache/")

# convert json labels from huggingface to yolo format
splits = real_ds.keys()
shutil.rmtree("datasets/scc_cell_detection_real", ignore_errors=True)
for split in splits:
    ds_dir = f"datasets/scc_cell_detection_real/{split}"
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

real_train_ds = sv.DetectionDataset.from_yolo("datasets/scc_cell_detection_real/train/images", "datasets/scc_cell_detection_real/train/labels", "datasets/scc_cell_detection_real/data.yaml")
real_val_ds = sv.DetectionDataset.from_yolo("datasets/scc_cell_detection_real/validation/images", "datasets/scc_cell_detection_real/validation/labels", "datasets/scc_cell_detection_real/data.yaml")
real_test_ds = sv.DetectionDataset.from_yolo("datasets/scc_cell_detection_real/test/images", "datasets/scc_cell_detection_real/test/labels", "datasets/scc_cell_detection_real/data.yaml")
print(f"Real dataset: {len(real_train_ds)} train images")
print(f"Real dataset: {len(real_val_ds)} validation images")
print(f"Real dataset: {len(real_test_ds)} test images")

NUM_TRAIN_IMAGES = 5000
REAL_10 = int(NUM_TRAIN_IMAGES * 0.9)
REAL_30 = int(NUM_TRAIN_IMAGES * 0.7)
REAL_50 = int(NUM_TRAIN_IMAGES * 0.5)
FAKE_10 = NUM_TRAIN_IMAGES - REAL_10
FAKE_30 = NUM_TRAIN_IMAGES - REAL_30
FAKE_50 = NUM_TRAIN_IMAGES - REAL_50

print(f"{REAL_10=} {FAKE_10=}")
print(f"{REAL_30=} {FAKE_30=}")
print(f"{REAL_50=} {FAKE_50=}")

scc_cell_detection_10 = sv.DetectionDataset.merge([sv.DetectionDataset(
                                                    classes=fake_ds.classes,
                                                    images={name: fake_ds.images[name] for name in list(fake_ds.images.keys())[:FAKE_10]},
                                                    annotations={name: fake_ds.annotations[name] for name in list(fake_ds.images.keys())[:FAKE_10]},
                                                ), 
                                                    sv.DetectionDataset(
                                                    classes=real_train_ds.classes,
                                                    images={name: real_train_ds.images[name] for name in list(real_train_ds.images.keys())[:REAL_10]},
                                                    annotations={name: real_train_ds.annotations[name] for name in list(real_train_ds.images.keys())[:REAL_10]},
                                                )])
scc_cell_detection_30 = sv.DetectionDataset.merge([sv.DetectionDataset(
                                                    classes=fake_ds.classes,
                                                    images={name: fake_ds.images[name] for name in list(fake_ds.images.keys())[:FAKE_30]},
                                                    annotations={name: fake_ds.annotations[name] for name in list(fake_ds.images.keys())[:FAKE_30]},
                                                    ), 
                                                    sv.DetectionDataset(
                                                    classes=real_train_ds.classes,
                                                    images={name: real_train_ds.images[name] for name in list(real_train_ds.images.keys())[:REAL_30]},
                                                    annotations={name: real_train_ds.annotations[name] for name in list(real_train_ds.images.keys())[:REAL_30]},
                                                )])
scc_cell_detection_50 = sv.DetectionDataset.merge([sv.DetectionDataset(
                                                    classes=fake_ds.classes,
                                                    images={name: fake_ds.images[name] for name in list(fake_ds.images.keys())[:FAKE_50]},
                                                    annotations={name: fake_ds.annotations[name] for name in list(fake_ds.images.keys())[:FAKE_50]},
                                                    ), 
                                                    sv.DetectionDataset(
                                                    classes=real_train_ds.classes,
                                                    images={name: real_train_ds.images[name] for name in list(real_train_ds.images.keys())[:REAL_50]},
                                                    annotations={name: real_train_ds.annotations[name] for name in list(real_train_ds.images.keys())[:REAL_50]},
                                                )])

scc_cell_detection_10.as_yolo("datasets/scc_cell_detection_10/train/images", "datasets/scc_cell_detection_10/train/labels", "datasets/scc_cell_detection_10/data.yaml")
scc_cell_detection_30.as_yolo("datasets/scc_cell_detection_30/train/images", "datasets/scc_cell_detection_30/train/labels", "datasets/scc_cell_detection_30/data.yaml")
scc_cell_detection_50.as_yolo("datasets/scc_cell_detection_50/train/images", "datasets/scc_cell_detection_50/train/labels", "datasets/scc_cell_detection_50/data.yaml")