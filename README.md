# Stable Diffusion 3.5 LoRA Fine-Tuning

This project provides a complete workflow for fine-tuning **Stable Diffusion 3.5 Medium** using **LoRA (Low-Rank Adaptation)**. Train on custom datasets with per-image prompts and generate high-quality images on consumer GPUs.

## Features

- **Stable Diffusion 3.5 Medium** - Latest model with excellent quality
- **LoRA Fine-tuning** - Efficient training without modifying base model
- **Per-image Prompts** - Custom prompts loaded from sidecar `.txt` files
- **Comparison Tools** - Side-by-side base vs LoRA model evaluation
- **Web Scraping** - Automated dataset collection with prompt file generation

## Prerequisites

- **Python 3.9+**
- **NVIDIA GPU** with 16GB+ VRAM (RTX 4080/4090, A6000, etc.)
- **CUDA 12.2+** installed
- **pipenv** for dependency management

## 1. Setup & Installation

First, set up the Python environment and install the required packages.

```bash

# Install dependencies
pipenv install

```

## Dataset Collection

### 1. Web Scraping
Use `webscraper.py` to download images and create empty prompt files:

```bash
pipenv run python webscraper.py
```

- Enter search query and number of images
- Images saved to `training-images/` with matching `.txt` files
- **Curate your dataset**: Remove low-quality/irrelevant images

### 2. Add Custom Prompts
Edit the `.txt` files to add descriptive prompts for each image:

```
training-images/
├── image1.jpg
├── image1.txt  ← "A woman with brown hair smiling"
├── image2.jpg
├── image2.txt  ← "A woman in a red dress outdoors"
└── ...
```

## 3. Fine-Tuning

Train your LoRA adapter with `finetune.py`:

```bash
pipenv run python finetune.py
```

Training will take some time, depending on your GPU and the number of images. The script will print the loss at each step. Once complete, the trained LoRA weights will be saved in the `lora_weights/` directory.

## Image Generation

### Standard Inference
Generate images with your fine-tuned model:

```bash
pipenv run python inference.py
```

### Model Comparison
Compare base model vs LoRA model side-by-side:

```bash
pipenv run python compare.py
```

Generates 4 comparison images:
- `base_baseline.png` - Base model + simple prompt
- `base_test.png` - Base model + your prompt
- `lora_baseline.png` - LoRA model + simple prompt  
- `lora_test.png` - LoRA model + your prompt


## Training Tips

1. **Quality over Quantity** - 20-50 high-quality images work better than hundreds of poor ones
2. **Diverse Prompts** - Use varied, descriptive prompts for each image
3. **Consistent Style** - Keep similar lighting/composition for style training
4. **Monitor Loss** - Use training loss and comparison script to refine fine-tuning

## Contributing

Feel free to submit issues and pull requests to improve the project!

## License

This project is open source. Please respect the licenses of the underlying models and libraries.
