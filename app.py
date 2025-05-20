import os
import base64
from io import BytesIO
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from demo import generate_floorplan, initialize_model

app = Flask(__name__)

# Initialize the TensorFlow model once when the app starts
print("Starting Flask app, initializing model...")
initialize_model()
print("Model initialized, Flask app is ready.")

def numpy_to_base64(image_array):
    """Convert a numpy array to base64 string for HTML display."""
    # Ensure the image is in uint8 format
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(image_array)
    
    # Convert to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return render_template('index.html', status='No image uploaded', error=True)
    
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', status='No image selected', error=True)
    
    try:
        # Read and convert the uploaded image to numpy array
        image = Image.open(file)
        image_np = np.array(image)
        
        # Generate floorplan
        original_image, floorplan_image = generate_floorplan(image_np)
        
        # Convert images to base64 for display
        original_b64 = numpy_to_base64(original_image)
        floorplan_b64 = numpy_to_base64(floorplan_image)
        
        return render_template('index.html',
                             original_image=original_b64,
                             floorplan_image=floorplan_b64,
                             status='Processing complete!')
    
    except Exception as e:
        return render_template('index.html',
                             status=f'Error processing image: {str(e)}',
                             error=True)

if __name__ == '__main__':
    app.run(debug=True) 