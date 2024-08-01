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
import torch.nn as nn
from torch.nn import init

from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile

class mAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.input_channel = 3
        self.num_output = num_classes
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels= 16, kernel_size= 11, stride= 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        init.xavier_uniform_(self.layer1[0].weight,gain= nn.init.calculate_gain('conv2d'))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 20, kernel_size= 5, stride= 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        init.xavier_uniform_(self.layer2[0].weight,gain= nn.init.calculate_gain('conv2d'))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels= 20, out_channels= 30, kernel_size= 3, stride= 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        init.xavier_uniform_(self.layer3[0].weight,gain= nn.init.calculate_gain('conv2d'))
       

        self.layer4 = nn.Sequential(
            nn.Linear(30*3*3, out_features=48),
            nn.ReLU(inplace=True)
        )
        init.kaiming_normal_(self.layer4[0].weight, mode='fan_in', nonlinearity='relu')

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=48, out_features=self.num_output)
        )
        init.kaiming_normal_(self.layer5[0].weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Squeezes or flattens the image, but keeps the batch dimension
        x = x.reshape(x.size(0), -1)
        x = self.layer4(x)
        logits= self.layer5(x)
        return logits


### BASIC TRAINING LOOP ###
import torch.optim as optim 

def train_model(model):
  criterion =  nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)

  for epoch in range(2):
    running_loss =0.0
    
    for i, data in enumerate(trainloader,0):
      
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 1000 == 999:
        print(f'[Ep: {epoch + 1}, Step: {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0
  
  return model

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

# a tuple of one or more example inputs are needed to trace the model
example_inputs = next(iter(trainloader))[0]

# Load float model
weights_path = EfficientnetLite0ModelFile.get_model_file_path()
model_to_quantize = EfficientNet.from_pretrained('efficientnet-lite0', 
                                     weights_path=weights_path,
                                     num_classes=10).to(device)
qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
model_to_quantize.train()
# prepare
model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs)

# training loop 
model_trained_prepared = train_model(model_prepared)

# quantize
model_quantized_trained = quantize_fx.convert_fx(model_trained_prepared)

# #### ALEXNET ####
# model_to_quantize = mAlexNet(num_classes= 10).to(device)
# qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
# model_to_quantize.train()
# # prepare
# model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs)

# # training loop 
# model_trained_prepared = train_model(model_prepared)

# # quantize
# model_quantized_trained = quantize_fx.convert_fx(model_trained_prepared)