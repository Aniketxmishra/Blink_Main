import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from blink.gnn_extractor import model_to_graph
from blink.gnn_model import ArchitectureGNN
import torchvision.models as models
try:
    from transformers import BertModel, RobertaModel, GPT2Model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed, skipping transformer models")

class SimpleCNN(nn.Module):
    def __init__(self, num_layers=3, channels=16):
        super(SimpleCNN, self).__init__()
        layers = []
        in_channels = 3
        for i in range(num_layers):
            out_channels = channels * (2 ** i)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(out_channels * (224 // (2**num_layers)) * (224 // (2**num_layers)), 10)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model_instance(model_name):
    # CNNs
    if model_name == "resnet18": return models.resnet18(weights=None)
    if model_name == "resnet50": return models.resnet50(weights=None)
    if model_name == "mobilenet_v2": return models.mobilenet_v2(weights=None)
    if model_name == "densenet121": return models.densenet121(weights=None)
    if model_name == "vgg16": return models.vgg16(weights=None)
    if model_name == "efficientnet_b0": return models.efficientnet_b0(weights=None)
    if model_name == "regnet_y_400mf": return models.regnet_y_400mf(weights=None)
    if model_name == "shufflenet_v2_x1_0": return models.shufflenet_v2_x1_0(weights=None)
    if model_name == "squeezenet1_0": return models.squeezenet1_0(weights=None)
    if model_name == "wide_resnet50_2": return models.wide_resnet50_2(weights=None)
    if model_name == "convnext_tiny": return models.convnext_tiny(weights=None)
    # New expanded set
    if model_name == "efficientnet_v2_s": return models.efficientnet_v2_s(weights=None)
    if model_name == "convnext_small": return models.convnext_small(weights=None)
    if model_name == "convnext_base": return models.convnext_base(weights=None)
    if model_name == "regnet_x_400mf": return models.regnet_x_400mf(weights=None)
    if model_name == "regnet_y_800mf": return models.regnet_y_800mf(weights=None)
    if model_name == "mnasnet1_0": return models.mnasnet1_0(weights=None)
    if model_name == "googlenet": return models.googlenet(weights=None)
    if model_name == "inception_v3": return models.inception_v3(weights=None)
    if model_name == "resnext50_32x4d": return models.resnext50_32x4d(weights=None)
    if model_name == "mobilenet_v3_large": return models.mobilenet_v3_large(weights=None)
    if model_name == "mobilenet_v3_small": return models.mobilenet_v3_small(weights=None)
    if model_name == "densenet169": return models.densenet169(weights=None)
    if model_name == "densenet201": return models.densenet201(weights=None)
    if model_name == "vgg19": return models.vgg19(weights=None)
    
    # Transformers (only if library is available)
    if HAS_TRANSFORMERS:
        if model_name == "bert-base": return BertModel.from_pretrained("bert-base-uncased")
        if model_name == "roberta-base": return RobertaModel.from_pretrained("roberta-base")
        if model_name == "gpt2": return GPT2Model.from_pretrained("gpt2")
    
    # Custom
    if model_name == "simple_cnn_3layers": return SimpleCNN(num_layers=3, channels=16)
    if model_name == "simple_cnn_5layers": return SimpleCNN(num_layers=5, channels=16)
    if model_name == "simple_cnn_3layers_wide": return SimpleCNN(num_layers=3, channels=32)
    
    raise ValueError(f"Unknown model name: {model_name}")

def get_model_family(model_name):
    if model_name in ["bert-base", "roberta-base", "gpt2"]:
        return "Transformer"
    all_cnns = ["resnet18", "resnet50", "mobilenet_v2", "densenet121", "vgg16", 
                "efficientnet_b0", "regnet_y_400mf", "shufflenet_v2_x1_0", "squeezenet1_0",
                "wide_resnet50_2", "convnext_tiny", "efficientnet_v2_s", "convnext_small",
                "convnext_base", "regnet_x_400mf", "regnet_y_800mf", "mnasnet1_0",
                "googlenet", "inception_v3", "resnext50_32x4d", "mobilenet_v3_large",
                "mobilenet_v3_small", "densenet169", "densenet201", "vgg19"]
    if "cnn" in model_name or model_name in all_cnns:
        return "CNN"
    else:
        return "Other"

class GNNDataset(Dataset):
    def __init__(self, dataframe):
        super(GNNDataset, self).__init__()
        self.df = dataframe.reset_index(drop=True)
        
        # Cache graph generations so we don't recreate them for different batch sizes of the same model
        self.graph_cache = {}
    
    def len(self):
        return len(self.df)
    
    def get(self, idx):
        row = self.df.iloc[idx]
        model_name = row['model_name']
        batch_size = float(row['batch_size'])
        exec_time = float(row['execution_time_ms'])
        memory = float(row['peak_memory_mb'])
        
        # Get graph
        if model_name not in self.graph_cache:
            try:
                model = get_model_instance(model_name)
                graph = model_to_graph(model)
                self.graph_cache[model_name] = graph
            except Exception as e:
                print(f"Error instantiating {model_name}: {e}")
                # Fallback dummy graph
                self.graph_cache[model_name] = Data(
                    x=torch.zeros((1, 12), dtype=torch.float32), 
                    edge_index=torch.empty((2, 0), dtype=torch.long)
                )
        
        graph = self.graph_cache[model_name]
        
        # We need a new Data object to attach the specific targets for this row
        data = Data(x=graph.x.clone(), edge_index=graph.edge_index.clone())
        data.bs = torch.tensor([[batch_size]], dtype=torch.float32)
        
        # Log space targets, avoiding negative infinity for 0s
        y = np.array([[np.log1p(exec_time), np.log1p(memory)]], dtype=np.float32)
        data.y = torch.tensor(y, dtype=torch.float32)
        
        return data

def train():
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Read Data
    csv_files = glob.glob('data/raw/*.csv')
    df_list = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Drop rows missing target metrics or where metrics are 0/None
    full_df = full_df.dropna(subset=['execution_time_ms', 'peak_memory_mb'])
    full_df = full_df[(full_df['execution_time_ms'] > 0) & (full_df['peak_memory_mb'] > 0)]
    
    # Filter noisy timings
    if 'timing_cv' in full_df.columns:
        initial_len = len(full_df)
        full_df = full_df[full_df['timing_cv'] <= 0.15]
        print(f"Dropped {initial_len - len(full_df)} rows due to timing_cv > 0.15")
    
    # Filter for unique models just to ensure we map properly
    full_df['family'] = full_df['model_name'].apply(get_model_family)
    
    # 2. Random row-level 80/20 split (NOT model-family holdout)
    # With only 135 rows and 17 models, model-family holdout creates an impossible task.
    # Random split allows the GNN to see different batch sizes of each architecture.
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # 3. Build Datasets
    print("Building datasets (this may take a moment to instantiate transformer models)...")
    train_dataset = GNNDataset(train_df)
    val_dataset = GNNDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 4. Model, Optimizer, Loss
    model = ArchitectureGNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.L1Loss()  # MAE Loss
    
    epochs = 100
    best_val_loss = float('inf')
    
    os.makedirs('models', exist_ok=True)
    
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch, batch.bs)
            
            # Calculate loss (in log space)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch.num_graphs
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch, batch.bs)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
        
        val_loss /= len(val_dataset)
        scheduler.step()
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/gnn_predictor.pth')
            
    print(f"Training complete! Best Val MAE: {best_val_loss:.4f}. Model saved to models/gnn_predictor.pth")

if __name__ == "__main__":
    train()
