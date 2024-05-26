import os
from tqdm import trange
from train_diffusion import Diffusion
from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler

BASE_MODEL_PATH = "/data/.cache/checkpoints"
SAMPLE_PATH = "/data/.cache/samples"

RUNS = {
    "scc_v2-10000-filtered-v1": [
        "epoch=179-step=225000.ckpt",
        "epoch=219-step=275000.ckpt",
        "epoch=319-step=400000.ckpt",
        "epoch=339-step=425000.ckpt",
        "epoch=349-step=437500.ckpt",
    ],
    "scc_v3-10000-filtered-v3": [
        "epoch=219-step=275000.ckpt",
        "epoch=299-step=375000.ckpt",
        "epoch=309-step=387500.ckpt",
        "epoch=329-step=412500.ckpt",
        "epoch=339-step=425000.ckpt",
    ],
    "scc_v3-10000-filtered-v4": [
        "epoch=269-step=337500.ckpt",
        "epoch=289-step=362500.ckpt",
        "epoch=299-step=375000-v1.ckpt",
        "epoch=319-step=400000.ckpt",
        "epoch=99-step=125000.ckpt",
    ],
    "scc_v4-10000-filtered-v2": [
        "epoch=159-step=200000.ckpt",
        "epoch=249-step=312500.ckpt",
        "epoch=269-step=337500.ckpt",
        "epoch=309-step=387500.ckpt",
        "epoch=329-step=412500.ckpt",
    ],
    "scc_v5-10000-filtered-v4": [
        "epoch=149-step=187500.ckpt",
        "epoch=189-step=237500.ckpt",
        "epoch=259-step=325000.ckpt",
        "epoch=279-step=350000.ckpt",
        "epoch=299-step=375000.ckpt",
    ],
}

SAMPLERS = {
    "DPM++ 2M": DPMSolverMultistepScheduler(solver_order=3, timestep_spacing="trailing"),
    "DPM++ 2M Karras": DPMSolverMultistepScheduler(solver_order=3, use_karras_sigmas=True, timestep_spacing="trailing"),
    "DPM++ 2M SDE": DPMSolverMultistepScheduler(solver_order=3, algorithm_type="sde-dpmsolver++", timestep_spacing="trailing"),
    "DPM++ 2M SDE Karras": DPMSolverMultistepScheduler(solver_order=3, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", timestep_spacing="trailing"),
    "DPM++ SDE": DPMSolverSinglestepScheduler(solver_order=3, timestep_spacing="trailing"),
    "DPM++ SDE Karras": DPMSolverSinglestepScheduler(solver_order=3, use_karras_sigmas=True, timestep_spacing="trailing"),
    "Euler": EulerDiscreteScheduler(timestep_spacing="trailing"),
    "Euler a": EulerAncestralDiscreteScheduler(timestep_spacing="trailing"),
    "DDIM": DDIMScheduler(timestep_spacing="trailing"),
}
STEPS_PER_SAMPLER = {
    "DPM++ 2M": [10, 15, 20, 25, 30, 35, 40, 45],
    "DPM++ 2M Karras": [10, 15, 20, 25, 30, 35, 40, 45],
    "DPM++ 2M SDE": [10, 15, 20, 25, 30, 35, 40, 45],
    "DPM++ 2M SDE Karras": [10, 15, 20, 25, 30, 35, 40, 45],
    "DPM++ SDE": [10, 15, 20, 25, 30, 35, 40, 45],
    "DPM++ SDE Karras": [10, 15, 20, 25, 30, 35, 40, 45],
    "Euler": [10, 15, 20, 25, 30, 35, 40, 45],
    "Euler a": [10, 15, 20, 25, 30, 35, 40, 45],
    "DDIM": [30, 35, 40, 45, 50, 55, 60, 65],
}

NUM_IMAGES = 8
BATCH_SIZE = 8

for run, models in RUNS.items():
    run_path = os.path.join(BASE_MODEL_PATH, run)
    for model in models:
        model_name = model.replace(".ckpt", "")
        model_path = os.path.join(run_path, model)
        model_dir = os.path.join(SAMPLE_PATH, run, model_name)
        for sampler_name, sampler in SAMPLERS.items():
            model_sample_path = os.path.join(model_dir, sampler_name)
            diffusion_model = Diffusion.load_from_checkpoint(model_path)
            diffusion_model.infer_scheduler = sampler
            for steps in STEPS_PER_SAMPLER[sampler_name]:
                steps_path = os.path.join(model_sample_path, f"{steps}")
                os.makedirs(steps_path, exist_ok=True)
                count = 0
                for i in trange(NUM_IMAGES // BATCH_SIZE, desc=f"{run}_{model_name}_{sampler_name}_{steps}"):
                    samples = diffusion_model.sample(batch_size=BATCH_SIZE, num_inference_steps=steps)
                    for img in samples:
                        img_name = os.path.join(steps_path, f"{run}_{model_name}_{sampler_name}_num_inf{steps}_{count:04d}.png")
                        img.save(img_name)
                        count += 1