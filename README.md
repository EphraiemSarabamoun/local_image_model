# Fine-Tuning Stable Diffusion with Custom Images

This project provides a complete workflow for fine-tuning a Stable Diffusion model on a custom dataset of images collected from the web. You can train the model to learn a specific person, object, or style, and then generate new images using your custom-trained model.

This guide will walk you through the three main stages:
1.  **Web Scraping**: Automatically download images for your dataset.
2.  **Fine-Tuning**: Train the model on your images using LoRA (Low-Rank Adaptation) for efficiency.
3.  **Inference**: Generate new images with your fine-tuned model.

## Prerequisites

- Python 3.8+
- An NVIDIA GPU with CUDA installed (for training and inference).
- `pipenv` for managing Python packages.

## 1. Setup & Installation

First, set up the Python environment and install the required packages.

```bash
# Install pipenv if you don't have it
pip install pipenv

# Install project dependencies from the Pipfile
pipenv install

```

## 2. Data Collection (Web Scraping)

Use the `webscraper.py` script to download images from the internet to use as your training data. These images will be saved to the `training-images/` directory.

1.  Run the script:
    ```bash
    pipenv run python webscraper.py
    ```

2.  You will be promped to enter the search terms you want to use be specific to get relevant images, and the number of images you want to download.

3.  Curate Your Dataset: After the script finishes, go into the `training-images/` folder and **delete any irrelevant, low-quality, or duplicate images**. A clean, high-quality dataset of 30-50 images is more effective than hundreds of poor-quality ones.

## 3. Fine-Tuning the Model

This step uses the `finetune.py` script to train a LoRA adapter on top of the base Stable Diffusion v1.5 model (use more powerful models if hardware allows). This process teaches the model about the concept in your training images.

Run the fine-tuning script:
    ```bash
    pipenv run python finetune.py
    ```

Training will take some time, depending on your GPU and the number of images. The script will print the loss at each step. Once complete, the trained LoRA weights will be saved in the `lora_weights/` directory.

## 4. Generating Images (Inference)

Now that you have your custom-trained LoRA weights, you can use `inference.py` to generate new images.

Run the inference script:
    ```bash
    pipenv run python inference.py
    ```

The script will prompt you to enter your text prompt. **You must include your trigger word (`sks`)** to activate the fine-tuned model.

    **Examples:**
    -   `a photo of sks person at the beach`
    -   `an oil painting of sks person`
    -   `sks person as a superhero, cinematic lighting`

The generated image will be saved to `output/generated_image.png`.

## File Descriptions

-   `webscraper.py`: Downloads images from DuckDuckGo based on your search queries.
-   `finetune.py`: Fine-tunes the Stable Diffusion model using the images in `training-images/` and saves the result to `lora_weights/`.
-   `inference.py`: Loads the base model and your LoRA weights to generate an image from a text prompt.
