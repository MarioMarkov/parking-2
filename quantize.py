import torch
import  torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import os 

import torchvision
model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)
#model_quantized.classifier[1] = nn.quantized.Linear(in_features=1280, out_features=2)

# scale=0.15309934318065643, 
# zero_point=75, 
# qscheme=torch.per_tensor_affine

print(model_quantized)
backend = "qnnpack"

# Non-quantized model
model = torch.load("models/full_256_alex_net_pk_lot.pth",map_location=torch.device('cpu'))
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

# # Post Training Static Quantization
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

# def print_model_size(mdl):
#     torch.save(mdl.state_dict(), "tmp.pt")
#     print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
#     os.remove('tmp.pt')

# print_model_size(model)
# print_model_size(model_static_quantized)
