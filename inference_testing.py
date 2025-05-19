import os
from inference import get_model
import supervision as sv
import cv2
from dotenv import load_dotenv

load_dotenv()

model = get_model(model_id="room-detection-6nzte/1", api_key=os.getenv("SUPERVISION_API_KEY"))

image = cv2.imread("demo/45765448.jpg")


# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)[0]

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results)

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
mask_annotator = sv.MaskAnnotator()

# annotate the image with our inference results
# annotated_image = bounding_box_annotator.annotate(
#     scene=image, detections=detections)
annotated_image = mask_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)
