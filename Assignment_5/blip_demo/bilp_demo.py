from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt

def generate_caption(image_path, text_prompt=None):
    """
    Generate a caption for an image using BLIP model
    :param image_path: path to image file or URL
    :param text_prompt: optional prompt to guide caption generation
    :return: generated caption
    """
    # Load the processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load the image
    if image_path.startswith('http'):
        image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')

    # Process the image
    if text_prompt:
        inputs = processor(image, text_prompt, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

def display_image_with_caption(image_path, caption):
    """Display the image with its generated caption"""
    img = Image.open(image_path) if not image_path.startswith('http') else Image.open(requests.get(image_path, stream=True).raw)
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, wrap=True)
    plt.show()

def main():
    print("BLIP Image Captioning Demo")
    print("-------------------------")
    
    # Get user input
    image_path = input("Enter image path or URL: ")
    text_prompt = input("(Optional) Enter a text prompt to guide captioning (or press Enter to skip): ")
    
    if not text_prompt:
        text_prompt = None
    
    # Generate and display caption
    caption = generate_caption(image_path, text_prompt)
    print("\nGenerated Caption:", caption)
    
    # Display the image with caption
    try:
        display_image_with_caption(image_path, caption)
    except Exception as e:
        print("Couldn't display image:", e)

if __name__ == "__main__":
    main()