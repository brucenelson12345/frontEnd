import nncf
import openvino.runtime as ov
import torch
from torchvision import datasets, transforms
from ultralytics import YOLO

# Instantiate your uncompressed model
model = YOLO('yolov8n.pt')

# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder("/home/pride/work/AIMET/yolov8/openVino/coco128/images/train2017", transform=transforms.Compose([transforms.ToTensor()]))
dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# Step 1: Initialize transformation function
def transform_fn(data_item):
    images, _ = data_item
    return images

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(model, calibration_dataset)