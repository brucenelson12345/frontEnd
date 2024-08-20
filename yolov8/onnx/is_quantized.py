import torch
from ultralytics import YOLO
import onnx, onnxruntime

def is_quantized(model):
    for module in model.modules():
        if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Conv2d, torch.quantization.QuantStub, torch.quantization.DeQuantStub)):
            return True
    return False

# Example usage
model = YOLO('yolov8n.pt')
#print(model)
if is_quantized(model):
    print("Model is quantized.")
else:
    print("Model is not quantized.")

model_int8 = 'models/static_quantized.onnx'
onnx_model = onnx.load(model_int8)
onnx.checker.check_model(onnx_model)
#print(onnx_model)

quantized = any(node.op_type.startswith("Q") for node in onnx_model.graph.node)
print("Is quantized:", quantized)

# Check data types of model inputs and outputs
for input in onnx_model.graph.input:
    print(input.type.tensor_type.elem_type)

for output in onnx_model.graph.output:
    print(output.type.tensor_type.elem_type)