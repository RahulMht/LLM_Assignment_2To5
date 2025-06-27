from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr
import os

def caption_image(image_path, text=""):
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Handle both file paths and Gradio's temporary file objects
        if isinstance(image_path, str):
            if image_path.startswith(('http://', 'https://')):
                # Handle URLs
                import requests
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                # Handle local file paths
                image = Image.open(image_path)
        else:
            # Handle Gradio file objects
            image = Image.fromarray(image_path)
        
        inputs = processor(image, text, return_tensors="pt") if text else processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## BLIP Image Captioning Demo")
    with gr.Row():
        img_input = gr.Image(type="filepath", label="Upload Image")
        text_input = gr.Textbox(label="Optional Prompt")
    btn = gr.Button("Generate Caption")
    output = gr.Textbox(label="Result")
    btn.click(caption_image, inputs=[img_input, text_input], outputs=output)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7885,
        share=True
    )