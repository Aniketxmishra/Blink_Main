import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from prediction_api import extract_model_features, predict_execution_time
from dynamic_gpu_predictor import DynamicGPUPredictor
from batch_size_optimizer import BatchSizeOptimizer

def recommend_batch_size(model_features):
    """Recommend optimal batch size based on model features"""
    total_params = model_features['total_parameters']
    model_size_mb = model_features['model_size_mb']
    
    if total_params > 100000000:  # Large models like VGG16 (138M) or RoBERTa (124M)
        recommended_batch = 2
        max_batch = 4
    elif total_params > 1000000:  # Medium models (1M+ parameters)
        recommended_batch = 4
        max_batch = 8
    else:  # Small models
        recommended_batch = 8
        max_batch = 16
    
    return {
        "recommended_batch_size": recommended_batch,
        "maximum_batch_size": max_batch,
        "reasoning": f"Based on model size ({model_size_mb:.2f} MB) and parameter count ({total_params:,})"
    }

def show_batch_optimization():
    """Show batch size optimization visualization"""
    st.subheader("Batch Size Optimization")
    
    model_options = ["simple_cnn_3layers", "simple_cnn_5layers", 
                    "simple_cnn_3layers_wide", "vgg16", "roberta-base"]
    selected_model = st.selectbox("Select Model", model_options)
    
    predictor = DynamicGPUPredictor()
    
    # Get model features
    if selected_model == "vgg16":
        model_features = {
            "total_parameters": 138357544,
            "trainable_parameters": 138357544,
            "model_size_mb": 527.7921447753906
        }
    else:
        model_features = {
            "total_parameters": 525354,
            "trainable_parameters": 525354,
            "model_size_mb": 2.004066467285156
        }
    
    # Get optimization results
    result = predictor.predict_and_optimize(model_features)
    recommendation = recommend_batch_size(model_features)
    
    # Create visualization
    batch_sizes = list(range(1, 33))
    exec_times = []
    throughputs = []
    
    for bs in batch_sizes:
        features = model_features.copy()
        features['batch_size'] = bs
        exec_time = predictor.predictor.predict(features)
        exec_times.append(exec_time)
        throughputs.append((bs * 1000) / exec_time)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=batch_sizes,
        y=exec_times,
        mode='lines+markers',
        name='Execution Time (ms)'
    ))
    
    fig.add_trace(go.Scatter(
        x=batch_sizes,
        y=throughputs,
        mode='lines+markers',
        name='Throughput (samples/s)',
        yaxis='y2'
    ))
    
    optimal_bs = result['optimal_batch_size']
    fig.add_vline(
        x=optimal_bs,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Optimal: {optimal_bs}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Batch Size Optimization for {selected_model}",
        xaxis_title="Batch Size",
        yaxis_title="Execution Time (ms)",
        yaxis2=dict(
            title="Throughput (samples/s)",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show recommendations
    st.info(f"""
    **Batch Size Recommendations:**
    - Recommended batch size: {recommendation['recommended_batch_size']}
    - Maximum batch size: {recommendation['maximum_batch_size']}
    - Reasoning: {recommendation['reasoning']}
    
    **Optimal Configuration:**
    - Optimal batch size: {result['optimal_batch_size']}
    - Predicted execution time: {result['predicted_execution_time']:.2f} ms
    - Estimated memory usage: {result['estimated_memory_usage']:.2f} MB
    """)

def show_model_comparison():
    """Show model comparison dashboard"""
    st.title("Model Architecture Comparison")
    
    # Sample model data for demonstration
    models = [
        {
            "name": "VGG16",
            "total_parameters": 138357544,
            "model_size_mb": 527.7921447753906
        },
        {
            "name": "Simple CNN",
            "total_parameters": 525354,
            "model_size_mb": 2.004066467285156
        }
    ]
    
    predictor = DynamicGPUPredictor()
    
    fig = go.Figure()
    
    for model in models:
        batch_sizes = list(range(1, 9))
        exec_times = []
        
        for bs in batch_sizes:
            features = {
                "total_parameters": model["total_parameters"],
                "trainable_parameters": model["total_parameters"],
                "model_size_mb": model["model_size_mb"],
                "batch_size": bs
            }
            exec_time = predictor.predictor.predict(features)
            exec_times.append(exec_time)
        
        fig.add_trace(go.Scatter(
            x=batch_sizes,
            y=exec_times,
            mode='lines+markers',
            name=model["name"]
        ))
    
    fig.update_layout(
        title="Execution Time Comparison",
        xaxis_title="Batch Size",
        yaxis_title="Execution Time (ms)",
        legend_title="Models"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="GPU Usage Prediction Dashboard",
        layout="wide"
    )
    
    st.title("GPU Usage Prediction Dashboard")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Batch Optimization", "Model Comparison"]
    )
    
    if page == "Batch Optimization":
        show_batch_optimization()
    else:
        show_model_comparison()

if __name__ == "__main__":
    main() 