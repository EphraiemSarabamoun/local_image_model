import torch
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler

# --- Configuration ---
base_model_path = "stabilityai/stable-diffusion-3.5-large"
lora_weights_path = "./lora_weights"
output_image_path = "./output/generated_image.png"

# Include the trigger word from training (e.g., 'sks person')
prompt = input("enter your prompt with sks to use the fine tuned models: ")

# --- Load Pipeline ---
pipe = StableDiffusion3Pipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    safety_checker=None  # Optional: Disable if outputs are being filtered incorrectly
)

# # --- Load LoRA Weights ---
# pipe.load_lora_weights(lora_weights_path)

# # --- Optimizations ---
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)  # Better scheduler for quality
# pipe.enable_attention_slicing()  # Memory efficiency
# pipe.enable_xformers_memory_efficient_attention()  # If xformers installed; speeds up and improves quality subtly

pipe.to("cuda")

# --- Generate Image ---
print(f"Generating image with prompt: {prompt}")
image = pipe(
    prompt,
    num_inference_steps=50,  # Increase for better detail
    guidance_scale=7.5,
).images[0]

# --- Save Image ---
image.save(output_image_path)
print(f"Image saved to {output_image_path}")