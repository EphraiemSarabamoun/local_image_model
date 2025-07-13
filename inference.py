import torch
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
from peft import PeftModel
import os

# --- Configuration ---
model_id = "stabilityai/stable-diffusion-3.5-medium"
lora_weights_path = "./lora_weights"
output_image_path = "./output/generated_image.png"

# Create output directory
os.makedirs("./output", exist_ok=True)

print("Loading quantized transformer...")
# Load the same quantized transformer as used in training
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load quantized transformer
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

print("Loading pipeline...")
# Load pipeline with quantized transformer
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    requires_safety_checker=False
)

print("Loading LoRA weights...")
# Load LoRA adapter properly
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_weights_path)

# Keep the default inference scheduler (DO NOT change to DDPM)
# The default scheduler is optimized for inference quality

# Apply optimizations
pipe.enable_attention_slicing()
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("XFormers enabled")
except:
    print("XFormers not available, skipping")

pipe.to("cuda")

# --- Generate Image ---
prompt = input("Enter your prompt: ")
print(f"Generating image with prompt: '{prompt}'")

# Generate with settings similar to debug script
image = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(42)  # For reproducibility
).images[0]

# --- Save Image ---
image.save(output_image_path)
print(f"Image saved to {output_image_path}")
