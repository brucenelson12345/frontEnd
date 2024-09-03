from PIL import Image
from qai_hub_models.models.yolov7 import Model as YOLOv7Model
from qai_hub_models.models.yolov7 import App as YOLOv7App
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.models.yolov7.demo import IMAGE_ADDRESS

# Load pre-trained model
torch_model = YOLOv7Model.from_pretrained()

# Load a simple PyTorch based application
app = YOLOv7App(torch_model)
image = load_image(IMAGE_ADDRESS, "yolov7")

# Perform prediction on a sample image
pred_image = app.predict(image)[0]
Image.fromarray(pred_image).show()