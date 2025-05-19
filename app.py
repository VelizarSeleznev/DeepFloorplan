import gradio as gr
from demo import generate_floorplan, initialize_model
import numpy as np

# Initialize the TensorFlow model once when the app starts
print("Starting Gradio app, initializing model...")
initialize_model()
print("Model initialized, Gradio app is ready.")

def floorplan_interface(input_image):
    """
    Gradio interface function.
    Takes a NumPy array (from Gradio image input) and returns two NumPy arrays.
    """
    if input_image is None:
        # Return blank images or a message if no image is provided
        # For simplicity, let's return blank 512x512 images
        blank_img = np.zeros((512, 512, 3), dtype=np.uint8)
        return blank_img, blank_img, "Please upload an image."

    # The input_image from gr.Image(type="numpy") is already a NumPy array (H, W, C)
    # and should be uint8.
    # generate_floorplan expects an RGB image.
    
    processed_input_image, floorplan_output_image = generate_floorplan(input_image)
    
    return processed_input_image, floorplan_output_image, "Processing complete."

# Define the Gradio interface
inputs = gr.Image(type="numpy", label="Upload Floorplan Image")
outputs = [
    gr.Image(type="numpy", label="Processed Input Image"),
    gr.Image(type="numpy", label="Generated Floorplan"),
    gr.Textbox(label="Status")
]

title = "DeepFloorplan: Automatic Floorplan Generation"
description = "Upload an image of a floorplan sketch to generate a structured floorplan."

# Launch the Gradio app
iface = gr.Interface(
    fn=floorplan_interface,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    live=False # Set to True for live updates as user changes input, False for a submit button.
)

if __name__ == '__main__':
    iface.launch() 