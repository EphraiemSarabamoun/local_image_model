import os
import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator  # For mixed precision and multi-GPU if needed

class CustomDataset(Dataset):
    def __init__(self, image_paths, prompt):
        self.image_paths = image_paths
        self.prompt = prompt
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Stable Diffusion normalization
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return {"pixel_values": image, "text": self.prompt}

# Initialize Accelerator for FP16/mixed precision
accelerator = Accelerator(mixed_precision="fp16")

# Load base Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(accelerator.device)
pipe.enable_attention_slicing()  # Memory optimization
pipe.vae.enable_xformers_memory_efficient_attention()  # Optional, if xformers installed

# Apply LoRA to UNet's cross-attention layers
lora_config = LoraConfig(
    r=16,  # Rank: higher = more capacity, but start low for small data
    lora_alpha=32,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Cross-attention targets
    lora_dropout=0.05,
)
pipe.unet = get_peft_model(pipe.unet, lora_config)

# Optionally apply LoRA to text encoder for better text understanding
# pipe.text_encoder = get_peft_model(pipe.text_encoder, lora_config)

# Prepare dataset to use all images in the training folder
training_dir = "training-images"
supported_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
image_paths = [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.lower().endswith(supported_extensions)]

prompt = "a photo of sks cat"  # Use 'sks' as unique token for your subject
dataset = CustomDataset(image_paths, prompt)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Small batch for small data

# Optimizer (only on LoRA params)
optimizer = AdamW(pipe.unet.parameters(), lr=1e-4)  # Add text_encoder.params if using

# Prepare with Accelerator
pipe.unet, optimizer, dataloader = accelerator.prepare(pipe.unet, optimizer, dataloader)

# Training loop
num_epochs = 50  # Adjust based on convergence; monitor loss
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        pixel_values = batch["pixel_values"].to(accelerator.device)
        prompts = batch["text"]

        # Encode text
        text_inputs = pipe.tokenizer(
            prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(accelerator.device)
        encoder_hidden_states = pipe.text_encoder(text_input_ids)[0]

        # Encode image to latents
        pixel_values = pixel_values.to(dtype=torch.float16)
        latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor

        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to latents
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise with UNet
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

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
accelerator.unwrap_model(pipe.unet).save_pretrained("lora_weights")
print("LoRA weights saved to 'lora_weights'")