import torch
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
from peft import PeftModel
import os

def load_base_model():
    """Load the base quantized model"""
    print("Loading base model...")
    
    # Load quantized transformer
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", 
        transformer=model_nf4,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    pipe.enable_attention_slicing()
    pipe.to("cuda")
    
    return pipe

def load_lora_model(base_pipe):
    """Load LoRA weights onto the base model"""
    print("Loading LoRA weights...")
    
    lora_weights_path = "./lora_weights"
    
    if not os.path.exists(lora_weights_path):
        print(f"ERROR: LoRA weights not found at {lora_weights_path}")
        return None
    
    # Create a copy of the base pipeline for LoRA
    lora_pipe = base_pipe
    lora_pipe.transformer = PeftModel.from_pretrained(base_pipe.transformer, lora_weights_path)
    print("LoRA weights loaded successfully")
    
    return lora_pipe

def generate_comparison_images():
    """Generate comparison images with both models and prompts"""
    
    # Define test prompts
    baseline_prompt = input("baseline prompt: ")  # Simple baseline prompt
    test_prompt = input("Enter your test prompt: ")
    
    # Generation settings
    num_steps = 50
    guidance_scale = 7.5
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    # Load base model
    base_pipe = load_base_model()

    # Test 1: Base model with baseline prompt
    print(f"1. Base model + Baseline prompt: '{baseline_prompt}'")
    image1 = base_pipe(
        baseline_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    image1.save("./output/base_baseline.png")
    
    # Test 2: Base model with test prompt
    print(f"2. Base model + Test prompt: '{test_prompt}'")
    image2 = base_pipe(
        test_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    image2.save("./output/base_test.png")


    # Load LoRA model
    lora_pipe = load_lora_model(base_pipe)
    
    if lora_pipe is None:
        return
    
    print("\n=== Generating Comparison Images ===")
    
    # Test 3: LoRA model with baseline prompt
    print(f"3. LoRA model + Baseline prompt: '{baseline_prompt}'")
    image3 = lora_pipe(
        baseline_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    image3.save("./output/lora_baseline.png")
    
    # Test 4: LoRA model with test prompt
    print(f"4. LoRA model + Test prompt: '{test_prompt}'")
    image4 = lora_pipe(
        test_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    image4.save("./output/lora_test.png")
    
    print("\n=== Comparison Complete ===")
    print("Generated images:")
    print("1. base_baseline.png - Base model with baseline prompt")
    print("2. base_test.png - Base model with test prompt")
    print("3. lora_baseline.png - LoRA model with baseline prompt")
    print("4. lora_test.png - LoRA model with test prompt")
    print("\nCompare these images to see:")
    print("- How LoRA affects the baseline prompt (1 vs 3)")
    print("- How LoRA affects your test prompt (2 vs 4)")
    print("- Overall LoRA training effectiveness")

if __name__ == "__main__":
    generate_comparison_images()
