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

device = "cuda"
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

# Copy model to qunatize
model_to_quantize = copy.deepcopy(model_fp).to(device)
model_to_quantize.eval()
qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)

# a tuple of one or more example inputs are needed to trace the model
example_inputs = next(iter(trainloader))[0]

# prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, 
                  example_inputs)
# no calibration needed when we only have dynamic/weight_only quantization
# quantize
model_quantized_dynamic = quantize_fx.convert_fx(model_prepared)

print_model_size(model_fp)
print_model_size(model_quantized_dynamic)