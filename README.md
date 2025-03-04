# Diffusion Based Brightfield Microscopy Image Generation
This repo contains the code to train unconditional diffusion models for synthetic brightfield microscopy image generation and training/evaluating state-of-the-art object detection models on datasets with varying proportions of synthetic images.

## filter_well_edge_app.py
A small tkinter utility that lazily displays all real brightfield microscopy images that I acquired for this thesis.
For each image the user code classifiy the image into either of two classes:
1. contains the well edge
2. does not contain the well edge

The results are saved to a `.jsonl` file to filter out images with the well edge.
This was necessary, since the thesis only dealt with images that not contain the well edge.
This dataset was then uploaded to huggingface for easier usage and distribution.
[Unfiltered Dataset](https://huggingface.co/datasets/mario-dg/brightfield-microscopy-scc) \
[Filtered Dataset](https://huggingface.co/datasets/mario-dg/brightfield-microscopy-scc-filtered)

## src/train_diffusion.py
This file contains the training script to train the unconditional diffusion models on the above
filtered brightfield microscopy images.
It uses [Hydra](https://github.com/facebookresearch/hydra.git) to configure the training setup, such as model architecture, dataset used and general hyperparameters.
[Huggingface Diffusers](https://github.com/huggingface/diffusers) for the diffusion model training and image generation and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for the reduction of boilerplate training code and ease of multi-gpu setup.

## src/test_diffusion.py
This file is responsible for generating images using the top 5 checkpoints of each trained model. It generates 8 images for each combination of checkpoint, inference steps, sampler and sampler configuration.
I evaluated the images to choose one final combination.

## src/inference_diffusion.py
The previously selected combination is used to generate the 10000 images and calculate the final FID score of the setup.

## create_detection_datasets.py
2500 images out of the 10000 were labeled with bounding boxes and then used to create the different datasets employed in this study.
Datasets with varying degree (10%, 30% and 50%) of generated images were assembled with the real data using this script.

## src/train_detection.py
The [Ultralytics](https://github.com/ultralytics/ultralytics) framework was used to train the object detection models (YOLOv8s, YOLOv8m, YOLOv8x, YOLOv9c, YOLOv9e, RT-DETR-l and RT-DETR-x).
This script trains each model on each dataset to create 28 fine-tuned models to compare.

## src/test_detection.py
This script leverages [Torchmetrics](https://github.com/Lightning-AI/torchmetrics) to calculate
the mAP at varying IoU threshold (50, 75 and 50:95).
These metrics server as the main argumentation point to answer the central research question.

## image_comparison.py
This script calculates the average brightness and contrast, aswell as a color bias for the generated and real microscopy images.
