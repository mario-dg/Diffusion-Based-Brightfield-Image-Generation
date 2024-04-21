import torch as th
from main import Diffusion
from diffusers.schedulers import DDIMScheduler
from tqdm import trange

CKPT_PATH = "/data/.cache/logs/checkpoints/usual-valley-60/pipeline-259/unet/diffusion_pytorch_model.safetensors"

map_location = {"cuda:1": "cuda:0"}
diffusion = Diffusion.load_from_checkpoint(CKPT_PATH, map_location=map_location, use_safetensors=True)

N = 128
B = 32

with diffusion.metrics():
    count = 0
    for _ in trange(N // B):
        samples = diffusion.sample(batch_size = B, num_inference_steps=25)
        for img in samples:
            img.save(f"/data/.cache/logs/checkpoints/usual-valley-60/samples/sample_{count:04d}_25.png")  
            count =+ 1  