import os
import torch
from diffusers import StableDiffusion3Pipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from PIL import Image
from bitsandbytes.optim import AdamW8bit
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator  # For mixed precision and multi-GPU if needed
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel

class CustomDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Stable Diffusion normalization
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
         img_path = self.image_paths[idx]
         image = Image.open(img_path).convert("RGB")
         image = self.transform(image)
         base, _ = os.path.splitext(img_path)
         txt_path = base + ".txt"
         with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.readline().strip()
         return {"pixel_values": image, "text": prompt}


model_id = "stabilityai/stable-diffusion-3.5-medium"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    requires_safety_checker=False
)

# For quantized models, use lighter CPU offloading to avoid conflicts
# pipe.enable_model_cpu_offload()  # Disabled - conflicts with quantization
# pipe.enable_sequential_cpu_offload()  # Disabled - conflicts with quantization

# Initialize Accelerator for mixed precision (without aggressive CPU offloading)
accelerator = Accelerator(
    mixed_precision="bf16",
    # cpu=True,  # Disabled - conflicts with quantized model
)

# Move pipeline to accelerator device after setup
pipe = pipe.to(accelerator.device)

# Memory optimizations that work with quantized models
pipe.enable_attention_slicing()  # Safe with quantization
pipe.vae.enable_slicing()  # Safe with quantization

# Set environment variable for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Replace inference scheduler with a training-compatible noise scheduler (DDPM)
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Apply LoRA to transformer's cross-attention layers
lora_config = LoraConfig(
    r=8,  # Rank: higher = more capacity, but start low for small data
    lora_alpha=32,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Cross-attention targets
    lora_dropout=0.05,
)
pipe.transformer = get_peft_model(pipe.transformer, lora_config)
pipe.transformer.enable_gradient_checkpointing()  # Uncomment to save memory during training

# Prepare dataset to use all images in the training folder
training_dir = "training-images"
supported_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
image_paths = [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.lower().endswith(supported_extensions)]
dataset = CustomDataset(image_paths)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Small batch for small data

# Optimizer (only on LoRA params)
optimizer_params = list(pipe.transformer.parameters())
# If using LoRA on text encoders: optimizer_params += list(pipe.text_encoder.parameters()) + list(pipe.text_encoder_2.parameters()) + list(pipe.text_encoder_3.parameters())
optimizer = AdamW8bit(optimizer_params, lr=1e-4)

# Prepare with Accelerator
pipe.transformer, optimizer, dataloader = accelerator.prepare(pipe.transformer, optimizer, dataloader)

# Training loop
num_epochs = 3  # Adjust based on convergence; monitor loss
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        pixel_values = batch["pixel_values"].to(accelerator.device)
        prompts = batch["text"]

        # Encode text using the pipeline's method (handles all encoders correctly)
        encoder_hidden_states, _, pooled_projections, _ = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            device=accelerator.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,  # No CFG during training
        )

        # Encode image to latents
        pixel_values = pixel_values.to(dtype=torch.bfloat16)
        latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor

        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise with transformer
        noise_pred = pipe.transformer(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections
        ).sample

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # Backprop and step
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

# Save LoRA weights
os.makedirs("lora_weights", exist_ok=True)
accelerator.unwrap_model(pipe.transformer).save_pretrained("lora_weights")
print("LoRA weights saved to 'lora_weights'")