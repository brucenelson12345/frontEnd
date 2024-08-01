import aimet_zoo_torch
from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile

weights_path = EfficientnetLite0ModelFile.get_model_file_path()
model = EfficientNet.from_pretrained('efficientnet-lite0', weights_path = weights_path)
