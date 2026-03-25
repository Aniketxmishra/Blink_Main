import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from diffusers import UNet2DModel
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from prediction_api import load_model, predict_for_custom_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_vit_model():
    """Create a Vision Transformer model"""
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072
    )
    model = ViTModel(config)
    return model

class SimpleGNN(nn.Module):
    """Create a simple Graph Neural Network"""
    def __init__(self, in_channels=3, hidden_channels=16, out_channels=10):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def create_diffusion_model():
    """Create a UNet-based diffusion model"""
    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )
    return model

class CNNTransformerHybrid(nn.Module):
    """Create a hybrid CNN-Transformer model"""
    def __init__(self, num_classes=10):
        super(CNNTransformerHybrid, self).__init__()
        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classification head
        self.fc = nn.Linear(64 * 56 * 56, num_classes)
        
    def forward(self, x):
        x = self.conv_layers(x)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (h*w, batch, channels)
        x = self.transformer(x)
        x = x.permute(1, 0, 2).flatten(1)  # (batch, h*w*channels)
        x = self.fc(x)
        return x

def profile_diverse_architectures():
    """Profile diverse model architectures and compare predictions"""
    # Load the prediction model
    prediction_model = load_model('models/gradient_boosting_model.joblib')
    
    # Define diverse model architectures
    diverse_models = {
        "vision_transformer_small": create_vit_model(),
        "simple_gnn": SimpleGNN(),
        "diffusion_unet": create_diffusion_model(),
        "cnn_transformer_hybrid": CNNTransformerHybrid()
    }
    
    # Batch sizes to test
    batch_sizes = [1, 2, 4]
    
    # Store results
    results = []
    
    # Test each model
    for name, model in diverse_models.items():
        print(f"\nTesting {name}...")
        
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
    results_df.to_csv('results/diverse_architectures_predictions.csv', index=False)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    sns.barplot(x='model_name', y='predicted_execution_time_ms', hue='batch_size', data=results_df)
    plt.title('Predicted Execution Time by Architecture Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/diverse_architectures_execution_time.png')
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='model_name', y='predicted_memory_usage_mb', hue='batch_size', data=results_df)
    plt.title('Predicted Memory Usage by Architecture Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/diverse_architectures_memory_usage.png')
    
    print(f"\nTesting complete. Results saved to results/diverse_architectures_predictions.csv")
    print(f"Visualizations saved to results/ directory")

if __name__ == "__main__":
    profile_diverse_architectures() 