# Preprocess onnx file
# python -m onnxruntime.quantization.preprocess --input yolov8n.onnx --output preprocessed.onnx
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'preprocessed.onnx'
model_int8 = 'dynamic_quantized.onnx'

# Quantize 
quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QUInt8)

