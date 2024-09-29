import torch
from diffusers import StableDiffusionPipeline

rand_seed = torch.manual_seed(42)
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 0.75
HEIGHT = 512
WIDTH = 512 

model_list = [
    "nota-ai/bk-sdm-small",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-xl-base-1.0",
]

def create_pipeline(model_name = model_list[1]):
    if torch.cuda.is_available():
        print("Using GPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors = True
        ).to('cuda')
    elif torch.backends.mps.is_available():
        print("Using MPS")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors = True
        ).to('mps')
    else:
        print("Using CPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors = True
        )
    return pipeline

def text2img(prompt, pipeline):
    images = pipeline(
        prompt, 
        guidance_scale = GUIDANCE_SCALE,
        num_inference_steps = NUM_INFERENCE_STEPS,
        generator = rand_seed,
        num_images_per_request = 1,
        height = HEIGHT,
        width = WIDTH
    ).images
    return images[0]
