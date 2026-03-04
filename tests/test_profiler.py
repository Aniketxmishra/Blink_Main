from model_profiler import ModelProfiler
import torch
import torch.nn as nn

# Define a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 56 * 56, 10)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.fc(x)
        return x

def main():
    # Create a simple model
    model = SimpleModel()
    
    # Create a model profiler
    profiler = ModelProfiler(save_dir='data/raw')
    
    # Profile the model with a small input shape and batch sizes
    results = profiler.profile_model(
        model=model,
        input_shape=(3, 224, 224),
        batch_sizes=[1, 2, 4],
        model_name="simple_model"
    )
    
    # Print the results
    print("\nProfiling Results:")
    print(results)

if __name__ == "__main__":
    main()
