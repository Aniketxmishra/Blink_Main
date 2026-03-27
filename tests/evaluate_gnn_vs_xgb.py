import warnings

import pandas as pd
from torchvision.models import mobilenet_v3_small, resnet18, shufflenet_v2_x1_0, squeezenet1_0

from prediction_api import (
    extract_model_features,
    load_model,
    predict_execution_time,
    predict_with_gnn,
)

warnings.filterwarnings("ignore")

models_to_test = {
    'ResNet18(Seen)': resnet18(weights=None),
    'ShuffleNetV2(Unseen)': shufflenet_v2_x1_0(weights=None),
    'SqueezeNet(Unseen)': squeezenet1_0(weights=None),
    'MobileNetV3(Unseen)': mobilenet_v3_small(weights=None)
}

xgb_pred_model, xgb_bounds, xgb_mem_model = load_model('models/gradient_boosting_model.joblib')

results = []
batch_sizes = [1, 4, 32]

for name, net in models_to_test.items():
    net.eval()
    features = extract_model_features(net, (3, 224, 224))
    xgb_preds = predict_execution_time(xgb_pred_model, {}, features, batch_sizes)
    
    for i, bs in enumerate(batch_sizes):
        gnn_pred = predict_with_gnn(net, bs)
        xgb_exec = xgb_preds[i]['predicted_execution_time_ms']
        gnn_exec = gnn_pred['exec_time_ms']
        
        results.append({
            'model': name,
            'batch_size': bs,
            'xgb_time_ms': xgb_exec,
            'gnn_time_ms': gnn_exec
        })

df = pd.DataFrame(results)
print(df)
df.to_csv('results/evaluation_gnn_vs_xgb.csv', index=False)
