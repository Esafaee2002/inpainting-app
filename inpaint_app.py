import numpy as np
import pandas as pd
import torch
import gradio as gr
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline

SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(DEVICE)
predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting"
)
pipe = pipe.to(DEVICE)

print(f"Models are loaded and running on {DEVICE}")

selected_pixels = []

def generate_mask(image, evt: gr.SelectData):
    selected_pixels.append(evt.index)
    predictor.set_image(np.array(image))
    input_points = np.array(selected_pixels)
    input_labels = np.ones(input_points.shape[0])
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )
    mask = masks[0]
    mask = np.logical_not(mask).astype(np.uint8) * 255
    return Image.fromarray(mask)

def inpaint(image, mask, prompt):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        image = image.resize((512, 512))
        mask = mask.resize((512, 512)).convert("L")

        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask
        ).images[0]

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Image.new("RGB", (512, 512), color=(255, 0, 0))

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input")
        mask_img = gr.Image(label="Mask")
        output_img = gr.Image(label="Output")

    with gr.Column():
        prompt_text = gr.Textbox(lines=1, label="Prompt")

    with gr.Row():
        submit_btn = gr.Button("Submit")

    input_img.select(generate_mask, inputs=[input_img], outputs=[mask_img])
    submit_btn.click(inpaint, inputs=[input_img, mask_img, prompt_text], outputs=[output_img])

if __name__ == "__main__":
    demo.launch()
