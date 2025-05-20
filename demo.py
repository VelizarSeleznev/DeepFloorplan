import os
import argparse
import numpy as np
import tensorflow as tf
import imageio.v2 as iio
from skimage.transform import resize as skimage_resize
from matplotlib import pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Ensure TF1 compatibility mode when using TF2
tf.compat.v1.disable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Global TensorFlow Variables ---
SESS = None
GRAPH = None
X_TENSOR = None
ROOM_TYPE_LOGIT_TENSOR = None
ROOM_BOUNDARY_LOGIT_TENSOR = None
MODEL_INITIALIZED = False
# --- End Global TensorFlow Variables ---

# color map
floorplan_map = {
	0: [255,255,255], # background
	1: [192,192,224], # closet
	2: [192,255,255], # batchroom/washroom
	3: [224,255,192], # livingroom/kitchen/dining room
	4: [255,224,128], # bedroom
	5: [255,160, 96], # hall
	6: [255,224,224], # balcony
	7: [255,255,255], # not used
	8: [255,255,255], # not used
	9: [255, 60,128], # door & window
	10:[  0,  0,  0]  # wall
}

def ind2rgb(ind_im, color_map=floorplan_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3), dtype=np.uint8) # Ensure uint8 for Gradio
	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb
	return rgb_im

def initialize_model():
	global SESS, GRAPH, X_TENSOR, ROOM_TYPE_LOGIT_TENSOR, ROOM_BOUNDARY_LOGIT_TENSOR, MODEL_INITIALIZED
	if MODEL_INITIALIZED:
		return

	print("Initializing TensorFlow model...")
	SESS = tf.compat.v1.Session()
	
	# initialize
	SESS.run(tf.group(tf.compat.v1.global_variables_initializer(),
						tf.compat.v1.local_variables_initializer()))

	# restore pretrained model
	saver = tf.compat.v1.train.import_meta_graph('./pretrained/pretrained_r3d/pretrained_r3d.meta', clear_devices=True)
	saver.restore(SESS, './pretrained/pretrained_r3d/pretrained_r3d')

	# get default graph
	GRAPH = tf.compat.v1.get_default_graph()

	# restore inputs & outpus tensor
	X_TENSOR = GRAPH.get_tensor_by_name('inputs:0')
	ROOM_TYPE_LOGIT_TENSOR = GRAPH.get_tensor_by_name('Cast:0')
	ROOM_BOUNDARY_LOGIT_TENSOR = GRAPH.get_tensor_by_name('Cast_1:0')
	MODEL_INITIALIZED = True
	print("TensorFlow model initialized.")

def generate_floorplan(input_image_np):
	global SESS, X_TENSOR, ROOM_TYPE_LOGIT_TENSOR, ROOM_BOUNDARY_LOGIT_TENSOR

	if not MODEL_INITIALIZED:
		initialize_model()

	# Store original image shape
	original_height, original_width = input_image_np.shape[0], input_image_np.shape[1]

	# Preprocess the input image (similar to main)
	# input_image_np is expected to be a uint8 RGB image
	if input_image_np.shape[2] == 4: # Handle RGBA from some uploads
		input_image_np = input_image_np[:,:,:3]

	# Ensure input_image_np is uint8 for consistent return type later
	if input_image_np.dtype != np.uint8:
		input_image_np = (input_image_np * 255).astype(np.uint8) if input_image_np.max() <= 1.0 else input_image_np.astype(np.uint8)


	im_resized_float64 = skimage_resize(input_image_np, (512, 512), anti_aliasing=True)
	im_processed = im_resized_float64.astype(np.float32) # Model expects float32 in [0,1]

	# infer results
	[room_type, room_boundary] = SESS.run([ROOM_TYPE_LOGIT_TENSOR, ROOM_BOUNDARY_LOGIT_TENSOR],
									feed_dict={X_TENSOR: im_processed.reshape(1,512,512,3)})
	room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

	# merge results
	floorplan_512 = room_type.copy()
	floorplan_512[room_boundary==1] = 9
	floorplan_512[room_boundary==2] = 10
	floorplan_rgb_512 = ind2rgb(floorplan_512) # This returns uint8, 512x512

	# Resize floorplan_rgb_512 to original image dimensions
	# skimage_resize output is float64 in [0,1] if input is uint8.
	# We need to convert it back to uint8.
	floorplan_rgb_resized_float = skimage_resize(floorplan_rgb_512, (original_height, original_width), anti_aliasing=True, preserve_range=True, order=0) # order=0 for nearest neighbor to preserve discrete colors
	floorplan_rgb_final = floorplan_rgb_resized_float.astype(np.uint8)


	return input_image_np, floorplan_rgb_final


def main_cli(args):
	# This function is for the command-line interface
	initialize_model() # Ensure model is loaded

	# load input image file
	im_uint8_from_file = iio.imread(args.im_path, pilmode='RGB')
	
	# Generate floorplan using the core logic
	# Note: generate_floorplan expects a NumPy array.
	# The generate_floorplan function handles its own resizing and type conversion.
	original_input_image, floorplan_rgb_output = generate_floorplan(im_uint8_from_file)

	# plot results for CLI
	plt.figure(figsize=(10, 5))
	plt.subplot(121)
	plt.imshow(original_input_image) # Show the original input image
	plt.title("Input Image")
	plt.axis('off')

	plt.subplot(122)
	plt.imshow(floorplan_rgb_output) # floorplan_rgb_output is already uint8
	plt.title("Generated Floorplan")
	plt.axis('off')
	
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	# Argument parsing for CLI
	parser = argparse.ArgumentParser(description="Generate a floorplan from an image.")
	parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
						help='Path to the input image file.')
	FLAGS, unparsed = parser.parse_known_args()
	main_cli(FLAGS)
