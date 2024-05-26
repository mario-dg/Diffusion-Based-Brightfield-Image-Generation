import torch as th
import random
from main import Diffusion
from diffusers import EulerAncestralDiscreteScheduler
from tqdm import trange

CKPT_PATH = "/data/.cache/checkpoints/scc_v3-10000-filtered-v4/epoch=319-step=400000.ckpt"

diffusion = Diffusion.load_from_checkpoint(CKPT_PATH)
diffusion.infer_scheduler = EulerAncestralDiscreteScheduler(timestep_spacing="trailing")

N = 10_000
B = 16

count = 1
with diffusion.metrics():
    for i in trange(N // B):
        inf_steps = random.randint(35, 40)
        samples = diffusion.sample(batch_size = B, num_inference_steps=inf_steps)
        diffusion.record_fake_data_for_FID(samples)
        for img in samples:
            img.save(f"/data/.cache/final_samples/sample_{count:05d}_{inf_steps}.png")  
            count += 1  
    print(f"Final FID metric: {diffusion.FID}")