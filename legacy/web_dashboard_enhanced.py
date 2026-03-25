import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
import torch
import psutil
import GPUtil
from prediction_api import extract_model_features, predict_execution_time
from dynamic_gpu_predictor import DynamicGPUPredictor
from batch_size_optimizer import BatchSizeOptimizer

# Utility Functions
def get_current_gpu_utilization():
    """Get current GPU utilization metrics"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Get first GPU
            return {
                'gpu_util': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            }
    except:
        return {
            'gpu_util': 0,
            'memory_used': 0,
            'memory_total': 0,
            'temperature': 0
        }

def create_timeseries_chart(data):
    """Create a time series chart from GPU metrics"""
    df = pd.DataFrame(data)
    df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq='S')
    return df.set_index('timestamp')

def create_interactive_charts(df):
    """Create interactive charts with brushing capability"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=("GPU Utilization", "Memory Usage", "Execution Time")
    )
    
    fig.add_trace(
        go.Scatter(x=df['batch_size'], y=df['gpu_utilization_percent'],
                  mode='lines+markers', name='GPU Utilization'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['batch_size'], y=df['memory_used_mb'],
                  mode='lines+markers', name='Memory Usage'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['batch_size'], y=df['execution_time_ms'],
                  mode='lines+markers', name='Execution Time'),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        dragmode='select',
        hovermode='x unified',
        selectdirection='h',
        hoverdistance=100
    )
    
    return fig

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

def set_theme_compatibility():
    """Set theme compatibility for light/dark modes"""
    is_dark_theme = st.sidebar.checkbox("Dark Theme", True)
    
    if is_dark_theme:
        st.markdown("""
        <style>
        :root {
            --background-color: #262730;
            --text-color: #FAFAFA;
            --card-background: #3B3F4B;
            --accent-color: #4CAF50;
        }
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        .css-1d391kg, .css-12oz5g7 {
            background-color: var(--card-background);
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        :root {
            --background-color: #FFFFFF;
            --text-color: #111111;
            --card-background: #F0F2F6;
            --accent-color: #4CAF50;
        }
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        .css-1d391kg, .css-12oz5g7 {
            background-color: var(--card-background);
        }
        </style>
        """, unsafe_allow_html=True)

# Dashboard Components
def show_realtime_monitor():
    """Show real-time GPU monitoring section"""
    st.title("Real-Time GPU Monitoring")
    
    chart_placeholder = st.empty()
    
    if st.button("Start Monitoring"):
        monitoring = True
        
        def update_data():
            data = []
            while monitoring:
                gpu_util = get_current_gpu_utilization()
                data.append(gpu_util)
                
                if len(data) > 100:
                    data = data[-100:]
                
                chart = create_timeseries_chart(data)
                chart_placeholder.line_chart(chart)
                
                time.sleep(1)
        
        thread = threading.Thread(target=update_data)
        thread.start()

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
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Execution Time", "Scaling Efficiency", "Parameter Efficiency"])
    
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
    
    with tab1:
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
        page_title="Enhanced GPU Usage Prediction Dashboard",
        layout="wide"
    )
    
    # Set theme
    set_theme_compatibility()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Real-Time Monitor", "Batch Optimization", "Model Comparison"]
    )
    
    if page == "Real-Time Monitor":
        show_realtime_monitor()
    elif page == "Batch Optimization":
        show_batch_optimization()
    else:
        show_model_comparison()

if __name__ == "__main__":
    main() 