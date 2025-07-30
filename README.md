# ImageSwap: Segment Anything + Diffusion Inpainting

A simple interactive application that combines [Meta AI's Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) with a 
[Stable Diffusion XL inpainting model](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) to enable **custom image editing**. 

This illustrates the potential of chaining together multiple powerful generative AI tools to create something more powerful than any one tool can achieve (similar, in spirit, to Agentic AI with LLMs).

Users can click on a region of an image to segment it, and then replace it with new content using a text prompt.

Built with [Gradio](https://www.gradio.app/).

---

## Features

- **Click-to-select segmentation** using SAM
- **Automatic mask generation** from selected points
- **Diffusion-based inpainting** to replace selected regions with custom content
- **Interactive UI** for image upload, mask preview, and final output

---

## Example Workflow

1. **Upload** an image
2. **Click** on the object or region to segment (SAM predicts the mask)
3. **Enter a prompt** for what you want to replace it with
4. **Click "Run Inpainting"** to generate the modified image
5. Optionally, reset and start again

---

## Tech Stack

- `Transformers` — for loading the Segment Anything model (SAM)
- `Diffusers` — to use the Stable Diffusion XL inpainting pipeline
- `Gradio` — for the web interface
- `PyTorch` — for model computation
- `PIL`, `NumPy` — for image manipulation

---

## Setup Instructions

### 1. Create a conda environment with the given yml file, then run imageswap.py !

