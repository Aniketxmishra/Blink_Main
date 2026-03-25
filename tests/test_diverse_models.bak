import torch
import torch.nn as nn
import torchvision.models as models
from prediction_api import load_model, predict_for_custom_model, create_sample_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_resnet_like_model(num_blocks=2, width_factor=1):
    """Create a simplified ResNet-like model"""
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
                
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = self.relu(out)
            return out
    
    class ResNetLike(nn.Module):
        def __init__(self, block, num_blocks, width_factor=1):
            super(ResNetLike, self).__init__()
            self.in_channels = 16 * width_factor
            
            self.conv1 = nn.Conv2d(3, 16 * width_factor, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16 * width_factor)
            self.relu = nn.ReLU(inplace=True)
            
            self.layer1 = self._make_layer(block, 16 * width_factor, num_blocks, stride=1)
            self.layer2 = self._make_layer(block, 32 * width_factor, num_blocks, stride=2)
            self.layer3 = self._make_layer(block, 64 * width_factor, num_blocks, stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64 * width_factor, 10)
            
        def _make_layer(self, block, out_channels, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_channels, out_channels, stride))
                self.in_channels = out_channels
            return nn.Sequential(*layers)
        
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out
    
    return ResNetLike(ResidualBlock, num_blocks, width_factor)

def create_transformer_like_model(num_layers=2, d_model=64):
    """Create a simplified Transformer-like model for image classification"""
    class TransformerLike(nn.Module):
        def __init__(self, num_layers=2, d_model=64):
            super(TransformerLike, self).__init__()
            
            # Initial convolutional layer to reduce spatial dimensions
            self.conv = nn.Conv2d(3, d_model, kernel_size=16, stride=16, padding=0)
            
            # Position embedding (simplified)
            self.pos_embedding = nn.Parameter(torch.randn(1, 196, d_model))
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Classification head
            self.classifier = nn.Linear(d_model, 10)
            
        def forward(self, x):
            # (B, 3, 224, 224) -> (B, d_model, 14, 14)
            x = self.conv(x)
            
            # (B, d_model, 14, 14) -> (B, d_model, 196) -> (B, 196, d_model)
            x = x.flatten(2).transpose(1, 2)
            
            # Add position embeddings
            x = x + self.pos_embedding
            
            # Apply transformer encoder
            x = self.transformer_encoder(x)
            
            # Global average pooling
            x = x.mean(dim=1)
            
            # Classification
            x = self.classifier(x)
            
            return x
    
    return TransformerLike(num_layers, d_model)

def test_diverse_models():
    """Test prediction accuracy on diverse model architectures"""
    # Load the prediction model
    prediction_model = load_model('models/gradient_boosting_model.joblib')
    
    # Define diverse model architectures to test
    test_models = {
        "simple_cnn_3layers": create_sample_model(3, 16),
        "simple_cnn_5layers": create_sample_model(5, 16),
        "simple_cnn_wide": create_sample_model(3, 32),
        "resnet_like_small": create_resnet_like_model(2, 1),
        "resnet_like_medium": create_resnet_like_model(3, 2),
        "transformer_like_small": create_transformer_like_model(2, 64),
        "transformer_like_medium": create_transformer_like_model(3, 128)
    }
    
    # Try to load pre-trained models if available
    try:
        test_models["resnet18"] = models.resnet18(weights=None)
        test_models["mobilenet_v2"] = models.mobilenet_v2(weights=None)
        test_models["efficientnet_b0"] = models.efficientnet_b0(weights=None)
    except:
        print("Some torchvision models couldn't be loaded. Continuing with custom models.")
    
    # Batch sizes to test
    batch_sizes = [1, 2, 4]
    
    # Store results
    results = []
    
    # Test each model
    for name, model in test_models.items():
        print(f"Testing {name}...")
        
        # Make predictions
        predictions = predict_for_custom_model(
            prediction_model, 
            model, 
            (3, 224, 224), 
            batch_sizes
        )
        
        # Store results
        for pred in predictions:
            results.append({
                "model_name": name,
                "batch_size": pred["batch_size"],
                "predicted_execution_time_ms": pred["predicted_execution_time_ms"],
                "predicted_memory_usage_mb": pred["predicted_memory_usage_mb"]
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/diverse_models_predictions.csv', index=False)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    sns.barplot(x='model_name', y='predicted_execution_time_ms', hue='batch_size', data=results_df)
    plt.title('Predicted Execution Time by Model and Batch Size')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/execution_time_comparison.png')
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='model_name', y='predicted_memory_usage_mb', hue='batch_size', data=results_df)
    plt.title('Predicted Memory Usage by Model and Batch Size')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/memory_usage_comparison.png')
    
    print(f"Testing complete. Results saved to results/diverse_models_predictions.csv")
    print(f"Visualizations saved to results/ directory")

if __name__ == "__main__":
    test_diverse_models()
