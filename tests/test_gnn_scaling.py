"""Quick test: does the retrained GNN properly differentiate batch sizes?"""
import torch
import numpy as np
from gnn_model import ArchitectureGNN
from gnn_extractor import model_to_graph
import torchvision.models as models

gnn = ArchitectureGNN()
gnn.load_state_dict(torch.load('models/gnn_predictor.pth', map_location='cpu', weights_only=True))
gnn.eval()

test_models = {
    'resnet18': models.resnet18(weights=None),
    'vgg16': models.vgg16(weights=None),
    'mobilenet_v2': models.mobilenet_v2(weights=None),
    'shufflenet_v2': models.shufflenet_v2_x1_0(weights=None),
}

lines = []
header = f"{'Model':<20} {'BS=1':>10} {'BS=4':>10} {'BS=16':>10} {'BS=32':>10} {'BS=64':>10} {'Ratio 64/1':>12}"
lines.append(header)
lines.append("-" * 85)

for name, net in test_models.items():
    graph = model_to_graph(net)
    preds = []
    for bs in [1, 4, 16, 32, 64]:
        bs_t = torch.tensor([[float(bs)]], dtype=torch.float32)
        with torch.no_grad():
            out = gnn(graph, bs_t)
        pred_ms = float(np.expm1(out[0, 0].item()))
        pred_ms = max(0.1, pred_ms)
        preds.append(pred_ms)
    ratio = preds[-1] / preds[0] if preds[0] > 0 else 0
    line = f"{name:<20} {preds[0]:>10.2f} {preds[1]:>10.2f} {preds[2]:>10.2f} {preds[3]:>10.2f} {preds[4]:>10.2f} {ratio:>12.2f}x"
    lines.append(line)

output = "\n".join(lines)
print(output)
with open('results/gnn_scaling_test.txt', 'w') as f:
    f.write(output)
