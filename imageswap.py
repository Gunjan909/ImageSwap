'''
Simple application that chains SAM with a diffusion model to enable custom image generation with a selected component of an original image 
'''

import gradio as gr
from PIL import Image
import numpy as np
import torch

# Define the optimal input size for the diffusion model.
# For Stable Diffusion XL, 1024x1024 is the common optimal size.
DIFFUSION_MODEL_TARGET_SIZE = (1024, 1024) # (width, height)

try:
    from transformers import SamModel, SamProcessor
    from diffusers import AutoPipelineForInpainting

    #load model and processor for SAM as well as stable diffusion based pipeline
    model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda" if torch.cuda.is_available() else "cpu")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipeline.enable_model_cpu_offload()

    #this function loads the points that were clicked on the image, and creates a mask out of them by calling SAM
    def get_processed_inputs_actual(image, input_points_list):
        flat_points = [point_set[0] for point_set in input_points_list]
        labels = [[1] * len(flat_points)]

        inputs = processor(
            images=image,
            input_points=[flat_points],
            input_labels=labels,
            return_tensors='pt'
        ).to(model.device)

        outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        best_mask = masks[0][0][outputs.iou_scores.argmax()]
        return ~best_mask.cpu().numpy()

    #takes a binary mask (0/1) and converts it to a green color , so we can visualize the mask generated so far
    def mask_to_rgb_actual(mask):
        bg_transparent = np.zeros(mask.shape + (4,), dtype=np.uint8)
        bg_transparent[mask == 1] = [0, 255, 0, 127]
        return bg_transparent

except Exception as e:
    print("Error in loading critical functions")
    print(e)


# Application state
state = {
    "image": None,
    "clicks": []
}

def on_click(current_image_pil: Image.Image, evt: gr.SelectData):
    """Handles clicks on the image display."""
    if current_image_pil is None:
        return "No image loaded to click on."
    
    x, y = evt.index
    state["clicks"].append([[x, y]])
    state["image"] = current_image_pil
    return f"Added point: ({x}, {y}) for SAM. Total points: {len(state['clicks'])}"

def get_mask_image_and_state():
    """Generates the mask using SAM and prepares it for display/diffusion.
       Returns only the display mask, and updates a hidden state for the binary mask."""
    if state["image"] is None or not state["clicks"]:
        return None, None # Return None for display mask and None for state
    try:
        binary_mask_np = get_processed_inputs_actual(state["image"], state["clicks"])
        rgba_mask_img = Image.fromarray(mask_to_rgb_actual(binary_mask_np), mode="RGBA")
        binary_mask_for_diffusion = (binary_mask_np > 0).astype(np.uint8) * 255
        
        # Return the RGBA image for display, and the binary mask for the state
        return rgba_mask_img, binary_mask_for_diffusion
    except Exception as e:
        print(f"Error generating mask: {e}")
        return None, None # Return None for both outputs on error

# run_inpainting to resize input image and mask, then resize output
def run_inpainting(prompt, neg_prompt, seed, binary_mask_input):
    """Runs the inpainting pipeline, resizing inputs and outputs."""
    if state["image"] is None or not state["clicks"] or binary_mask_input is None:
        return None
    
    rand_gen = torch.manual_seed(seed)
    
    # Store original dimensions for final resizing
    original_width, original_height = state["image"].size

    # Prepare image for pipeline (ensure it's PIL Image)
    image_for_pipeline = state["image"]
    if not isinstance(image_for_pipeline, Image.Image):
        image_for_pipeline = Image.fromarray(image_for_pipeline)

    # Convert binary mask (numpy array) to PIL Image
    mask_image_for_pipeline = Image.fromarray(binary_mask_input)

    # Resize inputs to diffusion model's optimal size ---
    resized_image = image_for_pipeline.resize(DIFFUSION_MODEL_TARGET_SIZE, Image.LANCZOS)
    
    # Resize the mask image (important to use the same resampling method)
    # Masks should be resized carefully to maintain their binary nature if necessary.
    # For inpainting, the mask should ideally be black/white.
    resized_mask = mask_image_for_pipeline.resize(DIFFUSION_MODEL_TARGET_SIZE, Image.LANCZOS)
    
    # Ensure the mask remains binary (0 or 255) after resizing
    resized_mask_np = np.array(resized_mask)
    resized_mask_np = (resized_mask_np > 127).astype(np.uint8) * 255 # Re-binarize
    resized_mask = Image.fromarray(resized_mask_np)


    try:
        result = pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=resized_image,      # Use the resized image
            mask_image=resized_mask,  # Use the resized mask
            guidance_scale=7,
            generator=rand_gen
        ).images[0]
        
        # Resize the output image back to the original input dimensions
        if result.size != (original_width, original_height):
            result = result.resize((original_width, original_height), Image.LANCZOS)
        
        return result
    except Exception as e:
        print(f"Error running inpainting pipeline: {e}")
        return None

def reset():
    """Resets the application state and UI components."""
    state["image"] = None
    state["clicks"] = []
    # Also reset the hidden binary mask state
    return None, None, None, None, "Upload an image to start."

# Gradio Interface Definition
with gr.Blocks() as demo:
    gr.Markdown("# ImageSwap: SAM + Diffusion Inpainting")
    gr.Markdown("## Instructions:")
    gr.Markdown("1. Click 'Upload Image' button to load base image into 'Original Image' display.")
    gr.Markdown("2. **Click directly on 'Original Image' display** to select points for SAM segmentation.")
    gr.Markdown("3. The 'Generated Mask' will update with each click, showing SAM's predicted segmentation.")
    gr.Markdown("4. Enter a text prompt and optional negative prompt for the new background.")
    gr.Markdown("5. Click 'Run Inpainting' to generate the final image.")

    # NEW: A hidden gr.State component to pass the binary mask
    binary_mask_state = gr.State(value=None) # Initialize with None

    with gr.Row():
        with gr.Column():
            image_display = gr.Image(
                label="Original Image",
                type="pil",
                interactive=True
            )
            upload_button = gr.UploadButton("Upload Image", file_types=["image"], label="Upload Image")
            click_info = gr.Textbox(label="Click Info", interactive=False)
        with gr.Column():
            mask_output = gr.Image(label="Generated Mask", type="pil", interactive=False)
        with gr.Column():
            inpaint_output = gr.Image(label="Inpainted Result", type="pil", interactive=False)

    with gr.Row():
        prompt = gr.Textbox(label="Prompt for New Background")
        neg_prompt = gr.Textbox(label="Negative Prompt (Optional)")
    
    with gr.Row():
        seed = gr.Number(label="Seed", value=1234, precision=0)
        run_button = gr.Button("Run Inpainting")
        reset_button = gr.Button("Reset")

    upload_button.upload(
        fn=lambda x: (x, None, "Image uploaded. Click on the image to select points."), # Also return None for binary_mask_state initially
        inputs=[upload_button],
        outputs=[image_display, binary_mask_state, click_info] # Update binary_mask_state here too
    ).then(
        fn=lambda: state.update(clicks=[], image=None),
        outputs=[]
    )

    image_display.select(
        fn=on_click,
        inputs=[image_display],
        outputs=[click_info]
    ).then(
        fn=get_mask_image_and_state, # Call the function that returns two outputs
        outputs=[mask_output, binary_mask_state] # Map outputs to mask_output and the new state
    )
    
    # Chain events for run button: get mask, then run inpainting
    run_button.click(
        fn=get_mask_image_and_state, # Ensure state is updated before running inpainting
        outputs=[mask_output, binary_mask_state], # Update both outputs
    ).then(
        fn=run_inpainting,
        inputs=[prompt, neg_prompt, seed, binary_mask_state], # Pass the hidden state as an input
        outputs=[inpaint_output]
    )

    reset_button.click(
        fn=reset,
        outputs=[image_display, mask_output, inpaint_output, binary_mask_state, click_info] # Reset the state too
    )

demo.launch()