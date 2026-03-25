import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from PIL import Image
import json
from prediction_api import extract_model_features, predict_execution_time

# Load the prediction model
@st.cache_resource
def load_prediction_model():
    model_path = 'models/gradient_boosting_model.joblib'
    return joblib.load(model_path)

# Create sample model architectures
def create_sample_model(num_layers=3, channels=16):
    """Create a sample CNN model for testing"""
    class SampleCNN(nn.Module):
        def __init__(self, num_layers=3, channels=16):
            super(SampleCNN, self).__init__()
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
    
    return SampleCNN(num_layers, channels)

# Load historical data
@st.cache_data
def load_historical_data():
    data_files = []
    for file in os.listdir('data/raw'):
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(f'data/raw/{file}')
                data_files.append(df)
            except:
                pass
    
    if data_files:
        return pd.concat(data_files, ignore_index=True)
    return pd.DataFrame()

# Main app
def main():
    st.set_page_config(page_title="GPU Usage Prediction Dashboard", layout="wide")
    
    # Sidebar
    st.sidebar.title("GPU Usage Prediction")
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Predict", "Historical Data", "Model Comparison", "About"])
    
    if page == "Predict":
        show_prediction_page()
    elif page == "Historical Data":
        show_historical_data()
    elif page == "Model Comparison":
        show_model_comparison()
    else:
        show_about_page()

def show_prediction_page():
    st.title("GPU Usage Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Custom CNN", "Pre-trained Models"]
        )
        
        if model_type == "Custom CNN":
            num_layers = st.slider("Number of Layers", 1, 10, 3)
            base_channels = st.slider("Base Channels", 8, 128, 16)
            
            # Create model description
            st.markdown(f"""
            **Model Architecture:**
            - Type: Custom CNN
            - Layers: {num_layers}
            - Base Channels: {base_channels}
            - Input Shape: (3, 224, 224)
            """)
            
            model = create_sample_model(num_layers, base_channels)
            
        else:
            model_name = st.selectbox(
                "Select Pre-trained Model",
                ["ResNet18", "ResNet50", "VGG16", "MobileNetV2", "DenseNet121"]
            )
            
            # Load selected model
            if model_name == "ResNet18":
                model = models.resnet18(weights=None)
            elif model_name == "ResNet50":
                model = models.resnet50(weights=None)
            elif model_name == "VGG16":
                model = models.vgg16(weights=None)
            elif model_name == "MobileNetV2":
                model = models.mobilenet_v2(weights=None)
            else:
                model = models.densenet121(weights=None)
        
        batch_sizes = st.multiselect(
            "Select Batch Sizes",
            [1, 2, 4, 8, 16, 32],
            default=[1, 2, 4]
        )
        
        if st.button("Predict GPU Usage"):
            with st.spinner("Calculating predictions..."):
                # Load prediction model
                prediction_model = load_prediction_model()
                
                # Extract features
                features = extract_model_features(model, (3, 224, 224))
                
                # Make predictions
                predictions = predict_execution_time(prediction_model, features, batch_sizes)
                
                # Display results
                st.subheader("Prediction Results")
                
                # Create DataFrame for display
                results_df = pd.DataFrame([
                    {
                        "Batch Size": p["batch_size"],
                        "Execution Time (ms)": p["predicted_execution_time_ms"]
                    } for p in predictions
                ])
                
                st.table(results_df)
                
                # Plot results
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x="Batch Size", y="Execution Time (ms)", data=results_df, ax=ax)
                ax.set_title("Predicted Execution Time by Batch Size")
                st.pyplot(fig)
                
                # Display model details
                total_params = sum(p.numel() for p in model.parameters())
                
                # Get model size in MB
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()
                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
                model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    with col2:
        st.subheader("Model Information")
        st.info("""
        This tool predicts GPU execution time for neural network models without actually running them on GPU hardware.
        
        **How to use:**
        1. Select a model type (Custom CNN or Pre-trained)
        2. Configure the model parameters
        3. Select batch sizes to predict for
        4. Click "Predict GPU Usage"
        
        The system will analyze the model architecture and predict execution times based on historical data from similar models.
        """)
        
        # Show sample architecture diagram
        st.subheader("Sample CNN Architecture")
        st.image("https://miro.medium.com/max/1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg", 
                 caption="Example CNN Architecture", 
                 use_column_width=True)

def show_historical_data():
    st.title("Historical Performance Data")
    
    # Load historical data
    df = load_historical_data()
    
    if df.empty:
        st.warning("No historical data found. Please check your data directory.")
        return
    
    # Data overview
    st.subheader("Data Overview")
    st.dataframe(df.head())
    
    # Model comparison
    st.subheader("Execution Time by Model and Batch Size")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    models = df['model_name'].unique()
    
    # Filter to common batch sizes
    common_batch_sizes = [1, 2, 4]
    filtered_df = df[df['batch_size'].isin(common_batch_sizes)]
    
    # Create pivot table
    pivot_df = filtered_df.pivot_table(
        index='model_name', 
        columns='batch_size', 
        values='execution_time_ms',
        aggfunc='mean'
    )
    
    # Plot heatmap
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title("Execution Time (ms) by Model and Batch Size")
    ax.set_ylabel("Model")
    ax.set_xlabel("Batch Size")
    st.pyplot(fig)
    
    # Model parameters comparison
    st.subheader("Model Size Comparison")
    
    # Get unique models and their sizes
    model_sizes = df.groupby('model_name')[['total_parameters', 'model_size_mb']].first()
    model_sizes = model_sizes.sort_values('total_parameters', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=model_sizes.index, y='model_size_mb', data=model_sizes.reset_index(), ax=ax)
    ax.set_title("Model Size (MB)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

def show_model_comparison():
    st.title("Model Architecture Comparison")
    
    # Load historical data
    df = load_historical_data()
    
    if df.empty:
        st.warning("No historical data found. Please check your data directory.")
        return
    
    # Select models to compare
    available_models = sorted(df['model_name'].unique())
    selected_models = st.multiselect(
        "Select models to compare",
        available_models,
        default=available_models[:3] if len(available_models) >= 3 else available_models
    )
    
    if not selected_models:
        st.warning("Please select at least one model to display.")
        return
    
    # Filter data for selected models
    filtered_df = df[df['model_name'].isin(selected_models)]
    
    # Execution time comparison
    st.subheader("Execution Time Comparison")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=filtered_df, 
        x='batch_size', 
        y='execution_time_ms', 
        hue='model_name',
        marker='o',
        ax=ax
    )
    ax.set_title("Execution Time vs Batch Size")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Execution Time (ms)")
    st.pyplot(fig)
    
    # Scaling efficiency
    st.subheader("Scaling Efficiency")
    
    # Calculate scaling efficiency (relative to batch size 1)
    pivot = filtered_df.pivot_table(
        index='model_name', 
        columns='batch_size', 
        values='execution_time_ms'
    )
    
    # Calculate relative scaling (normalized by batch size 1)
    for col in pivot.columns:
        if col > 1:
            pivot[f'scaling_{col}'] = pivot[col] / (pivot[1] * col)
    
    scaling_cols = [col for col in pivot.columns if isinstance(col, str) and col.startswith('scaling_')]
    
    if scaling_cols:
        scaling_df = pivot[scaling_cols].copy()
        scaling_df.columns = [f'Batch Size {col.split("_")[1]}' for col in scaling_cols]
        
        st.write("Scaling Efficiency (lower is better - indicates better parallelization)")
        st.dataframe(scaling_df)
        
        # Plot scaling efficiency
        scaling_df_plot = scaling_df.reset_index().melt(
            id_vars='model_name',
            var_name='batch_size',
            value_name='scaling_efficiency'
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=scaling_df_plot,
            x='model_name',
            y='scaling_efficiency',
            hue='batch_size',
            ax=ax
        )
        ax.set_title("Scaling Efficiency by Model (lower is better)")
        ax.set_ylabel("Relative Execution Time / Batch Size")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
    
    # Parameter efficiency
    st.subheader("Parameter Efficiency")
    
    # Get model parameters and execution time
    model_params = filtered_df.groupby('model_name')[['total_parameters', 'model_size_mb']].first()
    model_perf = filtered_df[filtered_df['batch_size'] == 1].groupby('model_name')['execution_time_ms'].first()
    
    efficiency_df = pd.DataFrame({
        'total_parameters': model_params['total_parameters'],
        'model_size_mb': model_params['model_size_mb'],
        'execution_time_ms': model_perf
    })
    
    efficiency_df['ms_per_million_params'] = efficiency_df['execution_time_ms'] / (efficiency_df['total_parameters'] / 1_000_000)
    
    st.write("Parameter Efficiency (ms per million parameters)")
    st.dataframe(efficiency_df[['total_parameters', 'execution_time_ms', 'ms_per_million_params']])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=efficiency_df.index,
        y='ms_per_million_params',
        data=efficiency_df,
        ax=ax
    )
    ax.set_title("Execution Time per Million Parameters (lower is better)")
    ax.set_ylabel("ms per Million Parameters")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

def show_about_page():
    st.title("About GPU Usage Prediction System")
    
    st.markdown("""
    ## Overview
    
    This web application provides GPU usage predictions for deep learning models without requiring actual execution on GPU hardware. It uses a machine learning model trained on performance data from various neural network architectures to predict execution times based on model characteristics.
    
    ## Features
    
    - **Prediction**: Estimate execution time for custom or pre-trained models across different batch sizes
    - **Historical Data**: View and analyze performance data from previously profiled models
    - **Model Comparison**: Compare execution times and scaling efficiency across different architectures
    
    ## How It Works
    
    1. The system extracts features from the neural network model (parameters, size, architecture details)
    2. These features are fed into a trained Gradient Boosting model
    3. The model predicts execution time based on patterns learned from historical data
    4. Results are displayed with visualizations for easy interpretation
    
    ## Dataset
    
    The prediction model was trained on data collected from various model architectures:
    
    - Simple CNNs with different configurations (3-layer, 5-layer variants)
    - Complex models like VGG16 (138M parameters)
    - Transformer models like RoBERTa-base (124M parameters)
    
    ## Accuracy
    
    When tested with ResNet18, the system achieved prediction errors under 6% for most batch sizes, demonstrating good generalization to unseen architectures.
    
    ## Future Improvements
    
    - Expanding to more diverse model architectures
    - Implementing tile-based execution modeling for improved accuracy
    - Adding power consumption and memory bandwidth predictions
    """)

if __name__ == "__main__":
    main()
