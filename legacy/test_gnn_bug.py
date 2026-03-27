import torch
from torchvision.models import resnet18, vgg16

from blink.gnn_extractor import model_to_graph
from blink.gnn_model import ArchitectureGNN

# Load model if exists, or randomly initialize
model = ArchitectureGNN()
try:
    model.load_state_dict(torch.load('models/gnn_predictor.pth', map_location='cpu'))
    print("Loaded pre-trained weights.")
except:
    print("Using untrained weights.")
model.eval()

res_model = resnet18()
data_res = model_to_graph(res_model)

vgg_model = vgg16()
data_vgg = model_to_graph(vgg_model)

print("--- ResNet18 ---")
for bs in [1, 32, 128]:
    bs_tensor = torch.tensor([[float(bs)]], dtype=torch.float32)
    out = model(data_res, bs_tensor)
    print(f"Batch Size {bs}: Output: {out.detach().numpy()}")

print("--- VGG16 ---")
for bs in [1, 32, 128]:
    bs_tensor = torch.tensor([[float(bs)]], dtype=torch.float32)
    out = model(data_vgg, bs_tensor)
    print(f"Batch Size {bs}: Output: {out.detach().numpy()}")
