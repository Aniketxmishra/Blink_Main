import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

from blink.feature_extractor import ModelFeatureExtractor


class SimpleCNN(nn.Module):
    def __init__(self, num_layers=3, channels=16):
        super().__init__()
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
    # Try dynamic loading for ALL torchvision models
    if hasattr(models, model_name):
        model_fn = getattr(models, model_name)
        input_shape = (3, 299, 299) if 'inception' in model_name else (3, 224, 224)
        try:
            return model_fn(weights=None), input_shape
        except TypeError:
            try:
                return model_fn(pretrained=False), input_shape
            except:
                return model_fn(), input_shape
        except Exception:
            pass

    
    # Custom CNNs
    if model_name == 'simple_cnn_3layers': return SimpleCNN(num_layers=3, channels=16), (3, 224, 224)
    if model_name == 'simple_cnn_5layers': return SimpleCNN(num_layers=5, channels=16), (3, 224, 224)
    if model_name == 'simple_cnn_3layers_wide': return SimpleCNN(num_layers=3, channels=32), (3, 224, 224)
    
    # Diverse
    try:
        if model_name == 'vision_transformer_small': 
            import diverse_architectures
            return diverse_architectures.create_vit_model(), (3, 224, 224)
        if model_name == 'diffusion_unet': 
            import diverse_architectures
            return diverse_architectures.create_diffusion_model(), (3, 32, 32)
        if model_name == 'cnn_transformer_hybrid': 
            import diverse_architectures
            return diverse_architectures.CNNTransformerHybrid(), (3, 224, 224)
        if model_name == 'simple_gnn':
            return None, None # skip for shape complexity
    except:
        pass
        
    # Transformers
    if model_name in ['bert-base', 'roberta-base', 'gpt2']:
        from transformers import AutoConfig, AutoModel
        try:
            if model_name == 'bert-base':
                config = AutoConfig.from_pretrained('bert-base-uncased')
                return AutoModel.from_config(config), (128,)
            elif model_name == 'roberta-base':
                config = AutoConfig.from_pretrained('roberta-base')
                return AutoModel.from_config(config), (128,)
            elif model_name == 'gpt2':
                config = AutoConfig.from_pretrained('gpt2')
                return AutoModel.from_config(config), (128,)
        except:
            pass

    return None, None

def main():
    print("Loading raw data...")
    all_data = []
    for f in glob.glob('data/raw/*.csv'):
        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except: pass
        
    if not all_data:
        print("No raw data found!")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Establish default hardware specs for older baseline data if missing
    if 'tflops_fp32' not in full_df.columns:
        full_df['tflops_fp32'] = np.nan
    if 'memory_bandwidth_gbps' not in full_df.columns:
        full_df['memory_bandwidth_gbps'] = np.nan
    if 'sm_count' not in full_df.columns:
        full_df['sm_count'] = np.nan
        
    full_df['tflops_fp32'] = full_df['tflops_fp32'].fillna(12.0)
    full_df['memory_bandwidth_gbps'] = full_df['memory_bandwidth_gbps'].fillna(336.0)
    full_df['sm_count'] = full_df['sm_count'].fillna(30.0)
    if 'gpu_name' not in full_df.columns:
        full_df['gpu_name'] = 'RTX Baseline'
    else:
        full_df['gpu_name'] = full_df['gpu_name'].fillna('RTX Baseline')
        
    unique_models = full_df['model_name'].unique()
    
    print(f"Found {len(unique_models)} unique models. Extracting rich features...")
    
    extractor = ModelFeatureExtractor(save_dir='data/enriched')
    enriched_features = {}
    
    for m in unique_models:
        print(f"Extracting features for {m}...")
        model, input_shape = get_model_instance(m)
        if model is None:
            print(f"Could not instantiate {m}, skipping feature extraction.")
            continue
            
        try:
            # Handle transformers which need LongTensor ids
            if 'bert' in m or 'roberta' in m or 'gpt2' in m:
                # Mock a custom model wrapper to extract flops with thop maybe?
                # Actually, extractor uses fake data based on input_shape
                # We'll skip complex ones or let them error out
                features = extractor.extract_model_features(model, input_shape, model_name=m)
            else:
                features = extractor.extract_model_features(model, input_shape, model_name=m)
                
            enriched_features[m] = features
        except Exception as e:
            print(f"Failed to extract features for {m}: {e}")
            
    # Now merge these rich features into full_df
    print(f"Successfully extracted features for {len(enriched_features)} models.")
    
    merged_rows = []
    for _, row in full_df.iterrows():
        m = row['model_name']
        if m in enriched_features:
            new_row = dict(row)
            # Add all rich features
            for k, v in enriched_features[m].items():
                if k not in new_row and k != 'layer_counts':
                    new_row[k] = v
            merged_rows.append(new_row)
            
    merged_df = pd.DataFrame(merged_rows)
    os.makedirs('data/enriched', exist_ok=True)
    merged_df.to_csv('data/enriched/enriched_data.csv', index=False)
    print("Saved enriched data to data/enriched/enriched_data.csv")

if __name__ == "__main__":
    main()
