import copy
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torchvision.models import resnet50
import timeit
fp32_model = resnet50().eval()
model = copy.deepcopy(fp32_model)
# `qconfig` means quantization configuration, it specifies how should we
# observe the activation and weight of an operator
# `qconfig_dict`, specifies the `qconfig` for each operator in the model
# we can specify `qconfig` for certain types of modules
# we can specify `qconfig` for a specific submodule in the model
# we can specify `qconfig` for some functioanl calls in the model
# we can also set `qconfig` to None to skip quantization for some operators
qconfig = get_default_qconfig("fbgemm")
qconfig_dict = {"": qconfig}
# `prepare_fx` inserts observers in the model based on the configuration in `qconfig_dict`
example_inputs = (torch.randn(1, 3, 224, 224),)
model_prepared = prepare_fx(model, qconfig_dict, example_inputs)
# calibration runs the model with some sample data, which allows observers to record the statistics of
# the activation and weigths of the operators
calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(100)]
for i in range(len(calibration_data)):
   model_prepared(calibration_data[i])
# `convert_fx` converts a calibrated model to a quantized model, this includes inserting
# quantize, dequantize operators to the model and swap floating point operators with quantized operators
model_quantized = convert_fx(copy.deepcopy(model_prepared))
# benchmark
x = torch.randn(1, 3, 224, 224)
# %timeit fp32_model(x)
# %timeit model_quantized(x)

file_path = "/tmp"

e_model = torch.export(model_quantized, x, "resnet_q")

