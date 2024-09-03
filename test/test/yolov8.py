from PIL import Image
from qai_hub_models.models.yolov8_det import Model as YOLOv8Model
from qai_hub_models.models.yolov8_det import App as YOLOv8App
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.models.yolov8_det.demo import IMAGE_ADDRESS

# Load pre-trained model
torch_model = YOLOv8Model.from_pretrained()

# Load a simple PyTorch based application
app = YOLOv8App(torch_model)
image = load_image(IMAGE_ADDRESS, "yolov8_det")

# Perform prediction on a sample image
pred_image = app.predict(image)[0]
Image.fromarray(pred_image).show()