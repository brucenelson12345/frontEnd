import os
from torchvision import transforms, datasets

DATASET_DIR = '/home/pride/work/datasets/ImageNet2012-download'   # Please replace this with a real directory

val_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

imagenet_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, 'val'), transform=val_transforms)

import random
from typing import Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
from aimet_torch.utils import in_eval_mode, get_device

EVAL_DATASET_SIZE = 500
CALIBRATION_DATASET_SIZE = 200

_datasets = {}

def _create_sampled_data_loader(dataset, num_samples):
    if num_samples not in _datasets:
        indices = random.sample(range(len(dataset)), num_samples)
        _datasets[num_samples] = Subset(dataset, indices)
    return DataLoader(_datasets[num_samples], batch_size=32)


def eval_callback(model: torch.nn.Module, num_samples: Optional[int] = None) -> float:
    if num_samples is None:
        num_samples = EVAL_DATASET_SIZE

    data_loader = _create_sampled_data_loader(imagenet_dataset, num_samples)
    device = get_device(model)
    
    correct = 0
    with in_eval_mode(model), torch.no_grad():
        for image, label in tqdm(data_loader):
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            top1 = logits.topk(k=1).indices
            correct += (top1 == label.view_as(top1)).sum()

    return int(correct) / num_samples

from torchvision.models import resnet18

model = resnet18(pretrained=True).eval()

if torch.cuda.is_available():
    model.to(torch.device('cuda'))

accuracy = eval_callback(model)
print(f'- FP32 accuracy: {accuracy}')

from aimet_torch.auto_quant import AutoQuant

class UnlabeledDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        images, _ = self._dataset[index]
        return images


unlabeled_imagenet_dataset = UnlabeledDatasetWrapper(imagenet_dataset)
unlabeled_imagenet_data_loader = _create_sampled_data_loader(unlabeled_imagenet_dataset,
                                                             CALIBRATION_DATASET_SIZE)

dummy_input = torch.randn((1, 3, 224, 224)).to(get_device(model))

auto_quant = AutoQuant(model,
                       dummy_input=dummy_input,
                       data_loader=unlabeled_imagenet_data_loader,
                       eval_callback=eval_callback)

sim, initial_accuracy = auto_quant.run_inference()
print(f"- Quantized Accuracy (before optimization): {initial_accuracy}")

from aimet_torch.adaround.adaround_weight import AdaroundParameters

ADAROUND_DATASET_SIZE = 200
adaround_data_loader = _create_sampled_data_loader(unlabeled_imagenet_dataset, ADAROUND_DATASET_SIZE)
adaround_params = AdaroundParameters(adaround_data_loader, num_batches=len(adaround_data_loader), default_num_iterations=2000)
auto_quant.set_adaround_params(adaround_params)

model, optimized_accuracy, encoding_path = auto_quant.optimize(allowed_accuracy_drop=0.01)
print(f"- Quantized Accuracy (after optimization):  {optimized_accuracy}")