import os
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets, models, transforms
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy

from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile


def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

device = "cpu"
print(device)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

batch_size = 8

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# Load float model
weights_path = EfficientnetLite0ModelFile.get_model_file_path()
model_fp = EfficientNet.from_pretrained('efficientnet-lite0', 
                                     weights_path=weights_path,
                                     num_classes=10).to(device)

model_fp.load_state_dict(torch.load("./float_model.pth", map_location=device))

# a tuple of one or more example inputs are needed to trace the model
example_inputs = next(iter(trainloader))[0]

# Copy model to qunatize
model_to_quantize = copy.deepcopy(model_fp)
qconfig_mapping = get_default_qconfig_mapping("qnnpack")
model_to_quantize.eval()
# prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
# # calibrate 
# with torch.no_grad():
#     for i in range(20):
#         batch = next(iter(trainloader))[0]
#         output = model_prepared(batch.to(device))

model_quantized_static = quantize_fx.convert_fx(model_prepared)

print_model_size(model_fp)
print_model_size(model_quantized_static)