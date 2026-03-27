import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class ArchitectureGNN(nn.Module):
    def __init__(self, node_feature_dim=12, hidden_dim=128):
        super().__init__()
        
        # 3 x GATConv layers
        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=4, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # MLP Head
        # The input to the MLP will be the pooled hidden_dim + 1 (for batch size target parameter)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Two outputs: exec_time_ms, memory_mb
        )

    def forward(self, data, batch_size_tensor):
        """
        data: torch_geometric.data.Data or Batch object
        batch_size_tensor: Tensor of shape [batch_size, 1] containing the batch size target.
        """
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        
        # For single graph inference without a DataLoader Batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # GATConv layers with ReLU
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Global mean pooling
        x_pooled = global_mean_pool(x, batch)
        
        # Ensure batch_size_tensor is shaped [batch_size, 1]
        if batch_size_tensor.dim() == 1:
            batch_size_tensor = batch_size_tensor.unsqueeze(1)
        
        # FIX: Normalize the batch size (it scales from 1 to 128+)
        batch_size_tensor = torch.log1p(batch_size_tensor)
            
        # Concat batch_size
        x_concat = torch.cat([x_pooled, batch_size_tensor.to(x_pooled.device)], dim=1)
        
        # MLP prediction
        out = self.mlp(x_concat)
        
        return out

if __name__ == '__main__':
    from torchvision.models import resnet18

    from .gnn_extractor import model_to_graph
    
    # Test creation
    model = ArchitectureGNN()
    print("Initiated ArchitectureGNN!")
    
    # Test forward pass with a dummy data object
    res_model = resnet18()
    data = model_to_graph(res_model)
    
    # Simulate a target query for a batch size of 32
    dummy_bs = torch.tensor([[32.0]], dtype=torch.float32)
    
    output = model(data, dummy_bs)
    print("Forward Pass Shape:", output.shape)
    print("Expected Shape:", torch.Size([1, 2]))
