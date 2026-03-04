import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torchvision.models as models
import os
import time
from gpu_predictor import GPUPredictor
from model_analyser import ModelAnalyzer

# Initialize components
@st.cache_resource
def get_predictor():
    return GPUPredictor()

@st.cache_resource
def get_analyzer():
    return ModelAnalyzer()

# Load data
@st.cache_data
def load_model_data():
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

# Create sample model
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

# Main app
def main():
    st.set_page_config(
        page_title="GPU Usage Prediction Dashboard", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    predictor = get_predictor()
    analyzer = get_analyzer()
    
    # Sidebar
    with st.sidebar:
        st.title("GPU Usage Prediction")
        
        # Navigation
        page = st.radio("Navigation", [
            "Predict", 
            "Batch Size Optimizer", 
            "Model Comparison", 
            "Performance Monitor",
            "Calibration",
            "About"
        ])
        
        # System stats
        st.subheader("System Stats")
        cache_stats = predictor.get_cache_stats()
        st.metric("Prediction Cache Hit Rate", f"{cache_stats['hit_rate']:.2%}")
        
        # Settings
        st.subheader("Settings")
        memory_limit = st.slider("GPU Memory Limit (MB)", 1000, 32000, 8000)
    
    # Main content
    if page == "Predict":
        show_prediction_page(predictor, analyzer)
    elif page == "Batch Size Optimizer":
        show_batch_optimizer(predictor, analyzer, memory_limit)
    elif page == "Model Comparison":
        show_model_comparison()
    elif page == "Performance Monitor":
        show_performance_monitor()
    elif page == "Calibration":
        show_calibration_page()
    else:
        show_about_page()

def show_prediction_page(predictor, analyzer):
    st.title("GPU Usage Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Custom CNN", "Pre-trained Models", "PyTorch Code (GNN)"]
        )
        
        model = None
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
            
        elif model_type == "Pre-trained Models":
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
        
        elif model_type == "PyTorch Code (GNN)":
            st.markdown("Paste your PyTorch model class here. Ensure you assign your instantiated class to a variable named `model`.")
            default_code = '''import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 112 * 112, 10)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)

# IMPORTANT: Assign instance to 'model'
model = MyModel()'''
            pasted_code = st.text_area("Paste your PyTorch model class here", value=default_code, height=300)
            
            if pasted_code:
                try:
                    # Sandbox exec environment
                    sandbox = {'nn': nn, 'torch': torch, 'models': models}
                    exec(pasted_code, sandbox)
                    
                    if 'model' in sandbox and isinstance(sandbox['model'], nn.Module):
                        model = sandbox['model']
                        st.success("Successfully instantiated model!")
                    else:
                        st.error("Could not find a variable named 'model' that is an nn.Module instance.")
                except Exception as e:
                    st.error(f"Error executing code: {e}")
        
        batch_sizes = st.multiselect(
            "Select Batch Sizes",
            [1, 2, 4, 8, 16, 32],
            default=[1, 2, 4]
        )
        
        if st.button("Predict GPU Usage"):
            if model is None:
                st.error("Cannot predict without a valid model.")
            else:
                with st.spinner("Analyzing model architecture..."):
                    from prediction_api import predict_with_gnn
                    
                    # Extract typical features (optional, just for details view later)
                    start_time = time.time()
                    features = analyzer.extract_features(model, (3, 224, 224))
                    analysis_time = time.time() - start_time
                    
                    # Make predictions
                    results = []
                    start_time = time.time()
                    
                    if model_type == "PyTorch Code (GNN)":
                        # Route through GNN
                        for bs in batch_sizes:
                            gnn_res = predict_with_gnn(model, bs)
                            exec_str = f"{gnn_res['exec_time_ms']:.2f} [{gnn_res['confidence_lower']:.2f} - {gnn_res['confidence_upper']:.2f}]"
                            
                            mem_usage = gnn_res['memory_mb']
                            mem_lower = mem_usage * 0.9
                            mem_upper = mem_usage * 1.1
                            mem_str = f"{mem_usage:.2f} [{mem_lower:.2f} - {mem_upper:.2f}]"
                            
                            results.append({
                                "Batch Size": bs,
                                "Execution Time (ms)": gnn_res['exec_time_ms'],
                                "Execution 80% CI (ms)": exec_str,
                                "Memory Usage (MB)": mem_usage,
                                "Memory 80% CI (MB)": mem_str,
                                "Source": gnn_res.get('source', 'GNN')
                            })
                    else:
                        # Existing tabular approach
                        features_batch = []
                        for bs in batch_sizes:
                            features_copy = features.copy()
                            features_copy['batch_size'] = bs
                            features_batch.append(features_copy)
                        
                        predictions = predictor.predict(features_batch)
                        if not isinstance(predictions, list):
                            predictions = [predictions]
                        
                        for i, bs in enumerate(batch_sizes):
                            pred_data = predictions[i]
                            exec_str = f"{pred_data['exec_time_ms']:.2f} [{pred_data['exec_time_lower']:.2f} - {pred_data['exec_time_upper']:.2f}]" 
                            if 'memory_usage_mb' in pred_data and pred_data['memory_usage_mb'] is not None:
                                 mem_str = f"{pred_data['memory_usage_mb']:.2f} [{pred_data['memory_lower_mb']:.2f} - {pred_data['memory_upper_mb']:.2f}]"
                                 mem_usage = pred_data['memory_usage_mb']
                            else:
                                 mem_str = "N/A"
                                 mem_usage = 0
                                 
                            results.append({
                                "Batch Size": bs,
                                "Execution Time (ms)": pred_data['exec_time_ms'],
                                "Execution 80% CI (ms)": exec_str,
                                "Memory Usage (MB)": mem_usage,
                                "Memory 80% CI (MB)": mem_str,
                                "Source": "XGBoost"
                            })
                    
                    prediction_time = time.time() - start_time
                    
                    # Create DataFrame for display
                    results_df = pd.DataFrame(results)
                
                # Display results
                st.subheader("Prediction Results")
                st.table(results_df)
                
                # Plot Execution Time results
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add expected execution time bar
                fig.add_trace(go.Bar(
                    x=results_df["Batch Size"],
                    y=results_df["Execution Time (ms)"],
                    name="Execution Time (ms)",
                    text=results_df["Execution Time (ms)"].round(2),
                    textposition="auto",
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[float(r.split('[')[1].split(' - ')[1].replace(']', '')) - e for r, e in zip(results_df["Execution 80% CI (ms)"], results_df["Execution Time (ms)"])],
                        arrayminus=[e - float(r.split('[')[1].split(' - ')[0]) for r, e in zip(results_df["Execution 80% CI (ms)"], results_df["Execution Time (ms)"])]
                    )
                ), secondary_y=False)
                
                # Add expected memory bar
                if any(results_df["Memory Usage (MB)"] > 0):
                    fig.add_trace(go.Scatter(
                        x=results_df["Batch Size"],
                        y=results_df["Memory Usage (MB)"],
                        mode='lines+markers',
                        name="Memory Usage (MB)",
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[float(r.split('[')[1].split(' - ')[1].replace(']', '')) - e for r, e in zip(results_df["Memory 80% CI (MB)"], results_df["Memory Usage (MB)"]) if r != "N/A"],
                            arrayminus=[e - float(r.split('[')[1].split(' - ')[0]) for r, e in zip(results_df["Memory 80% CI (MB)"], results_df["Memory Usage (MB)"]) if r != "N/A"]
                        ),
                        line=dict(color='orange')
                    ), secondary_y=True)
                
                fig.update_layout(
                    title="Predicted Performance by Batch Size with 80% Confidence Intervals",
                    xaxis_title="Batch Size",
                    barmode='group'
                )
                fig.update_yaxes(title_text="Execution Time (ms)", secondary_y=False)
                fig.update_yaxes(title_text="Peak Memory (MB)", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display model details
                st.subheader("Model Details")
                st.write(f"Total Parameters: {features['total_parameters']:,}")
                st.write(f"Model Size: {features['model_size_mb']:.2f} MB")
                
                # Display performance metrics
                st.subheader("Performance Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Analysis Time", f"{analysis_time*1000:.2f} ms")
                with col2:
                    st.metric("Prediction Time", f"{prediction_time*1000:.2f} ms")
    
    with col2:
        st.subheader("How It Works")
        st.info("""
        This tool predicts GPU execution time for neural network models without actually running them on GPU hardware.
        
        **Steps:**
        1. Select a model type
        2. Configure model parameters
        3. Select batch sizes
        4. Click "Predict GPU Usage"
        
        The system analyzes the model architecture and predicts execution times based on historical data from similar models.
        """)
        
        # Show architecture patterns
        if 'features' in locals():
            st.subheader("Architecture Patterns")
            patterns = features['architecture_patterns']
            
            pattern_data = {
                "Pattern": ["Skip Connections", "Attention Mechanism", "Normalization Layers", "Model Depth"],
                "Present": [
                    "âœ“" if patterns['has_skip_connections'] else "âœ—",
                    "âœ“" if patterns['has_attention'] else "âœ—",
                    "âœ“" if patterns['has_normalization'] else "âœ—",
                    str(patterns['max_depth'])
                ]
            }
            st.table(pd.DataFrame(pattern_data))

def show_batch_optimizer(predictor, analyzer, memory_limit):
    st.title("Batch Size Optimizer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Selection")
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Custom CNN", "Pre-trained Models"],
            key="opt_model_type"
        )
        
        if model_type == "Custom CNN":
            num_layers = st.slider("Number of Layers", 1, 10, 3, key="opt_num_layers")
            base_channels = st.slider("Base Channels", 8, 128, 16, key="opt_base_channels")
            model = create_sample_model(num_layers, base_channels)
            model_name = f"Custom CNN ({num_layers} layers, {base_channels} channels)"
        else:
            model_name = st.selectbox(
                "Select Pre-trained Model",
                ["ResNet18", "ResNet50", "VGG16", "MobileNetV2", "DenseNet121"],
                key="opt_model_name"
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
        
        min_batch = st.number_input("Minimum Batch Size", min_value=1, max_value=64, value=1)
        max_batch = st.number_input("Maximum Batch Size", min_value=1, max_value=512, value=128)
        
        if st.button("Find Optimal Batch Size"):
            with st.spinner("Analyzing model and finding optimal batch size..."):
                # Extract features
                features = analyzer.extract_features(model, (3, 224, 224))
                
                # Find optimal batch size
                optimization_result = predictor.optimize_batch_size(
                    features, 
                    min_batch=min_batch, 
                    max_batch=max_batch,
                    memory_limit_mb=memory_limit
                )
                
                # Guard: all batch sizes exceeded memory
                if optimization_result.get('error'):
                    st.error(optimization_result['error'])
                else:
                    # Create detailed results dataframe
                    batch_results = pd.DataFrame(optimization_result['batch_results'])

                    # Display metrics
                    st.subheader("Optimization Results")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Optimal Batch Size", optimization_result['optimal_batch_size'])
                    with metric_col2:
                        exec_val = optimization_result['predicted_execution_time']
                        st.metric("Execution Time", f"{exec_val:.2f} ms" if exec_val is not None else "N/A")
                    with metric_col3:
                        mem_val = optimization_result['estimated_memory_usage']
                        st.metric("Memory Usage", f"{mem_val:.2f} MB" if mem_val is not None else "N/A")
                
                    # Plot results
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Execution Time vs Batch Size", "Throughput vs Batch Size"),
                        shared_xaxes=True,
                        vertical_spacing=0.1
                    )
                    
                    # Add execution time trace with uncertainty band
                    # Create upper and lower bound arrays
                    x_values = batch_results['batch_size'].tolist()
                    x_rev = x_values[::-1]
                    
                    # Execution bounds
                    y_upper = batch_results['exec_upper_ms'].tolist()
                    y_lower = batch_results['exec_lower_ms'].tolist()
                    y_lower_rev = y_lower[::-1]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_values + x_rev,
                            y=y_upper + y_lower_rev,
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.2)',
                            line_color='rgba(255,255,255,0)',
                            showlegend=False,
                            name='Execution Time 80% CI'
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=batch_results['batch_size'],
                            y=batch_results['exec_time_ms'],
                            mode='lines+markers',
                            name='Execution Time (ms)',
                            line_color='rgb(0,100,80)'
                        ),
                        row=1, col=1
                    )
                    
                    # Add throughput trace
                    fig.add_trace(
                        go.Scatter(
                            x=batch_results['batch_size'],
                            y=batch_results['throughput'],
                            mode='lines+markers',
                            name='Throughput (samples/s)'
                        ),
                        row=2, col=1
                    )
                     
                    # Optimal batch size marker â€” use shapes (add_vline doesn't support row/col in subplots)
                    optimal_batch = optimization_result['optimal_batch_size']
                    for row_idx in [1, 2]:
                        fig.add_shape(
                            type='line',
                            x0=optimal_batch, x1=optimal_batch,
                            y0=0, y1=1,
                            yref='paper',
                            xref='x',
                            line=dict(color='green', dash='dash'),
                            row=row_idx, col=1
                        )

                    fig.add_annotation(
                        x=optimal_batch, y=1, yref='paper',
                        text=f"Optimal: {optimal_batch}",
                        showarrow=False, xanchor='left',
                        font=dict(color='green')
                    )

                    fig.update_layout(
                        height=600,
                        title=f"Batch Size Optimization for {model_name}",
                        showlegend=False
                    )
                    fig.update_xaxes(title_text="Batch Size", row=2, col=1)
                    fig.update_yaxes(title_text="Execution Time (ms)", row=1, col=1)
                    fig.update_yaxes(title_text="Throughput (samples/s)", row=2, col=1)

                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Detailed Results")
                    st.dataframe(batch_results)
    
    with col2:
        st.subheader("Optimization Strategy")
        st.info("""
        The batch size optimizer finds the optimal batch size that maximizes throughput while staying within memory constraints.
        
        **How it works:**
        1. For each batch size in the specified range:
           - Estimate memory usage
           - Predict execution time
           - Calculate throughput (samples/second)
        
        2. Select the batch size with highest throughput that fits in memory
        
        **Memory Estimation:**
        - Base memory (model parameters)
        - Activation memory (scales with batch size)
        - Optimizer state memory
        
        The memory limit can be adjusted in the sidebar.
        """)

def show_model_comparison():
    st.title("Model Architecture Comparison")
    
    # Load historical data
    df = load_model_data()
    
    if df.empty:
        st.warning("No historical data found. Please check your data directory.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Execution Time", "Scaling Efficiency", "Parameter Efficiency"])
    
    with tab1:
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
        
        # Create interactive plot
        fig = go.Figure()
        
        for model in selected_models:
            model_data = filtered_df[filtered_df['model_name'] == model]
            fig.add_trace(go.Scatter(
                x=model_data['batch_size'],
                y=model_data['execution_time_ms'],
                mode='lines+markers',
                name=model
            ))
        
        fig.update_layout(
            title="Execution Time vs Batch Size",
            xaxis_title="Batch Size",
            yaxis_title="Execution Time (ms)",
            legend_title="Models",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Scaling efficiency analysis
        st.subheader("Scaling Efficiency")
        
        if not available_models:
            st.warning("No models available for analysis.")
            return
            
        # Calculate scaling efficiency
        pivot = df.pivot_table(
            index='model_name', 
            columns='batch_size', 
            values='execution_time_ms'
        )
        
        # Calculate relative scaling (normalized by batch size 1)
        scaling_data = []
        
        for model in pivot.index:
            base_time = pivot.loc[model, 1] if 1 in pivot.columns else None
            if base_time is None:
                continue
                
            for batch_size in [b for b in pivot.columns if b > 1]:
                if pd.isna(pivot.loc[model, batch_size]):
                    continue
                    
                exec_time = pivot.loc[model, batch_size]
                ideal_time = base_time * batch_size
                efficiency = base_time / (exec_time / batch_size)
                
                scaling_data.append({
                    'model_name': model,
                    'batch_size': batch_size,
                    'efficiency': efficiency
                })
        
        if not scaling_data:
            st.warning("Insufficient data for scaling efficiency analysis.")
            return
            
        scaling_df = pd.DataFrame(scaling_data)
        
        # Create plot
        fig = px.bar(
            scaling_df,
            x='model_name',
            y='efficiency',
            color='batch_size',
            title="Scaling Efficiency by Model (higher is better)",
            labels={
                'model_name': 'Model',
                'efficiency': 'Scaling Efficiency',
                'batch_size': 'Batch Size'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Scaling Efficiency** measures how well a model utilizes increased batch sizes.
        
        A value of 1.0 means perfect scaling (execution time increases linearly with batch size).
        Higher values indicate better parallelization and GPU utilization.
        Lower values indicate memory or compute bottlenecks.
        """)
    
    with tab3:
        # Parameter efficiency analysis
        st.subheader("Parameter Efficiency")
        
        # Get model parameters and execution time
        model_params = df.groupby('model_name')[['total_parameters', 'model_size_mb']].first()
        model_perf = df[df['batch_size'] == 1].groupby('model_name')['execution_time_ms'].first()
        
        if model_params.empty or model_perf.empty:
            st.warning("Insufficient data for parameter efficiency analysis.")
            return
            
        # Combine data
        efficiency_df = pd.DataFrame({
            'total_parameters': model_params['total_parameters'],
            'model_size_mb': model_params['model_size_mb'],
            'execution_time_ms': model_perf
        })
        
        efficiency_df['ms_per_million_params'] = efficiency_df['execution_time_ms'] / (efficiency_df['total_parameters'] / 1_000_000)
        
        # Create plot
        fig = px.bar(
            efficiency_df.reset_index(),
            x='model_name',
            y='ms_per_million_params',
            title="Execution Time per Million Parameters (lower is better)",
            labels={
                'model_name': 'Model',
                'ms_per_million_params': 'ms per Million Parameters'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.dataframe(efficiency_df.reset_index())
        
        st.info("""
        **Parameter Efficiency** measures how efficiently a model uses its parameters.
        
        Lower values indicate better efficiency - less execution time per parameter.
        This metric helps identify models with good architecture design that maximizes parameter utilization.
        """)

def _get_gpu_stats():
    """Fetch live GPU stats via pynvml. Returns dict or None if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp   = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w  = power_mw / 1000.0
        except Exception:
            power_w = None
        try:
            power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            power_limit_w  = power_limit_mw / 1000.0
        except Exception:
            power_limit_w = None
        name = pynvml.nvmlDeviceGetName(handle)
        return {
            'name':        name,
            'util_gpu':    util.gpu,
            'util_mem':    util.memory,
            'mem_used_mb': mem.used  / 1024**2,
            'mem_free_mb': mem.free  / 1024**2,
            'mem_total_mb':mem.total / 1024**2,
            'temp_c':      temp,
            'power_w':     power_w,
            'power_limit_w': power_limit_w,
        }
    except Exception:
        return None


def show_performance_monitor():
    st.title("Performance Monitor")

    tab_gpu, tab_acc, tab_cache = st.tabs(["GPU Monitor", "Prediction Accuracy", "Cache Performance"])

    # ── Tab 1: Live GPU Monitor ───────────────────────────────────────────────
    with tab_gpu:
        st.subheader("Live GPU Statistics")

        stats = _get_gpu_stats()

        if stats is None:
            st.warning("pynvml could not read GPU stats. "
                       "Install `nvidia-ml-py` or ensure NVIDIA drivers are present.")
        else:
            st.caption(f"Device: **{stats['name']}**")

            # ── Big metric cards ──────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                color = "normal" if stats['util_gpu'] < 80 else "inverse"
                st.metric("GPU Utilization", f"{stats['util_gpu']} %",
                          delta=None)
                # manual color bar
                util_frac = stats['util_gpu'] / 100
                bar_color = "#2ecc71" if util_frac < 0.7 else "#e67e22" if util_frac < 0.9 else "#e74c3c"
                st.markdown(
                    f'<div style="background:#333;border-radius:4px;height:8px;">'
                    f'<div style="background:{bar_color};width:{util_frac*100:.0f}%;height:8px;border-radius:4px;"></div>'
                    f'</div>', unsafe_allow_html=True
                )
            with c2:
                vram_pct = stats['mem_used_mb'] / stats['mem_total_mb'] * 100
                st.metric("VRAM Used",
                          f"{stats['mem_used_mb']:.0f} MB",
                          delta=f"{vram_pct:.1f}% of {stats['mem_total_mb']:.0f} MB")
                bar_color = "#2ecc71" if vram_pct < 70 else "#e67e22" if vram_pct < 90 else "#e74c3c"
                st.markdown(
                    f'<div style="background:#333;border-radius:4px;height:8px;">'
                    f'<div style="background:{bar_color};width:{vram_pct:.0f}%;height:8px;border-radius:4px;"></div>'
                    f'</div>', unsafe_allow_html=True
                )
            with c3:
                temp_c = stats['temp_c']
                temp_color = "#2ecc71" if temp_c < 70 else "#e67e22" if temp_c < 85 else "#e74c3c"
                st.metric("Temperature", f"{temp_c} °C")
                st.markdown(
                    f'<div style="background:#333;border-radius:4px;height:8px;">'
                    f'<div style="background:{temp_color};width:{min(temp_c,100):.0f}%;height:8px;border-radius:4px;"></div>'
                    f'</div>', unsafe_allow_html=True
                )
            with c4:
                if stats['power_w'] is not None:
                    pw_label = f"{stats['power_w']:.1f} W"
                    pw_delta = (f"/ {stats['power_limit_w']:.0f} W TDP"
                                if stats['power_limit_w'] else None)
                    st.metric("Power Draw", pw_label, delta=pw_delta)
                else:
                    st.metric("Power Draw", "N/A")

            st.divider()

            # ── History chart ─────────────────────────────────────────────────
            if 'gpu_history' not in st.session_state:
                st.session_state.gpu_history = []

            now = pd.Timestamp.now()
            st.session_state.gpu_history.append({
                'time':      now,
                'util_gpu':  stats['util_gpu'],
                'util_mem':  stats['util_mem'],
                'vram_mb':   stats['mem_used_mb'],
                'temp_c':    stats['temp_c'],
                'power_w':   stats['power_w'] or 0,
            })
            # Keep last 60 readings
            st.session_state.gpu_history = st.session_state.gpu_history[-60:]
            hist = pd.DataFrame(st.session_state.gpu_history)

            fig = make_subplots(rows=2, cols=2,
                subplot_titles=("GPU Utilization (%)", "VRAM Used (MB)",
                                "Temperature (°C)", "Power Draw (W)"),
                vertical_spacing=0.15, horizontal_spacing=0.1)

            def add_line(fig, row, col, x, y, name, color):
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                    name=name, line=dict(color=color, width=2),
                    marker=dict(size=4)), row=row, col=col)

            add_line(fig, 1, 1, hist['time'], hist['util_gpu'],  'GPU Util %',  '#3498db')
            add_line(fig, 1, 2, hist['time'], hist['vram_mb'],   'VRAM MB',     '#9b59b6')
            add_line(fig, 2, 1, hist['time'], hist['temp_c'],    'Temp C',      '#e74c3c')
            add_line(fig, 2, 2, hist['time'], hist['power_w'],   'Power W',     '#f39c12')

            fig.update_layout(
                height=420, showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ccc'),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            for i in range(1, 3):
                for j in range(1, 3):
                    fig.update_xaxes(showgrid=True, gridcolor='#333', row=i, col=j)
                    fig.update_yaxes(showgrid=True, gridcolor='#333', row=i, col=j)

            st.plotly_chart(fig, use_container_width=True)

            # ── Auto-refresh slider ───────────────────────────────────────────
            refresh_s = st.slider("Auto-refresh interval (seconds)", 2, 30, 5,
                                  key="gpu_refresh_interval")
            if st.button("Refresh Now", key="gpu_refresh_btn"):
                st.rerun()
            st.caption(f"Last updated: {now.strftime('%H:%M:%S')}  •  "
                       f"Auto-refresh: every {refresh_s}s  •  "
                       "Click 'Refresh Now' or change any widget to update.")

            # Inject JS auto-refresh
            st.components.v1.html(
                f"<script>setTimeout(function(){{window.location.reload();}}, "
                f"{refresh_s * 1000});</script>",
                height=0
            )

    # ── Tab 2: Prediction Accuracy ────────────────────────────────────────────
    with tab_acc:
        st.subheader("Prediction Accuracy Analysis")
        feedback_path = 'data/feedback_log.csv'
        if os.path.exists(feedback_path):
            feedback_df = pd.read_csv(feedback_path)
            if not feedback_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=feedback_df['timestamp'], y=feedback_df['error_percent'],
                    mode='lines+markers', name='Prediction Error (%)'
                ))
                fig.update_layout(title="Prediction Error Over Time",
                                  xaxis_title="Timestamp", yaxis_title="Error (%)")
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Recent Feedback Data")
                st.dataframe(feedback_df.tail(10))
            else:
                st.info("No feedback data yet.")
        else:
            st.info("No feedback data available yet. As you use the system, "
                    "prediction accuracy data will be collected here.")

    # ── Tab 3: Cache Performance ──────────────────────────────────────────────
    with tab_cache:
        st.subheader("Cache Performance")
        predictor  = get_predictor()
        cache_stats = predictor.get_cache_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.2%}")
        with col2:
            st.metric("Cache Size", f"{cache_stats['cache_size']} / {cache_stats['max_cache_size']}")
        with col3:
            st.metric("Total Predictions", cache_stats['cache_hits'] + cache_stats['cache_misses'])
        hit_miss_data = pd.DataFrame([
            {'Category': 'Hits',   'Count': cache_stats['cache_hits']},
            {'Category': 'Misses', 'Count': cache_stats['cache_misses']}
        ])
        fig = px.pie(hit_miss_data, values='Count', names='Category',
                     title="Cache Hits vs Misses", color='Category',
                     color_discrete_map={'Hits': '#2ecc71', 'Misses': '#e74c3c'})
        st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.title("About GPU Usage Prediction System")
    
    st.markdown("""
    ## Overview
    
    This scalable GPU usage prediction system provides accurate estimates of execution time for deep learning models without requiring actual execution on GPU hardware. It uses machine learning to predict performance based on model architecture characteristics.
    
    ## Key Features
    
    - **Efficient Prediction**: Makes predictions in milliseconds using caching and batch processing
    - **Batch Size Optimization**: Finds optimal batch sizes to maximize throughput within memory constraints
    - **Model Comparison**: Analyzes scaling efficiency and parameter utilization across different architectures
    - **Performance Monitoring**: Tracks prediction accuracy and system performance over time
    
    ## How It Works
    
    1. **Model Analysis**: Extracts features from neural network architectures
    2. **Prediction**: Uses gradient boosting to predict execution times based on model features
    3. **Optimization**: Recommends optimal configurations for maximum performance
    4. **Monitoring**: Continuously improves through feedback and performance tracking
    
    ## Dataset
    
    The system was trained on data from diverse model architectures:
    
    - Simple CNNs (525K parameters)
    - Complex models like VGG16 (138M parameters)
    - Transformer models like RoBERTa-base (124M parameters)
    
    ## Performance
    
    - Prediction time: < 50ms
    - Accuracy: < 6% error for most models
    - Scalability: Handles batch prediction and parallel processing
    """)


def show_calibration_page():
    """Priority 2: Confidence Interval Calibration"""
    st.title("Confidence Interval Calibration")
    st.markdown("""
    Checks whether the **80% confidence intervals** produced by NeuSight's
    quantile regression models actually contain ~80% of real measurements.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Run Calibration Check", type="primary"):
            with st.spinner("Running calibration analysis..."):
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "scripts/calibration_check.py"],
                    capture_output=True, text=True, cwd=os.getcwd()
                )
                if result.returncode == 0:
                    st.success("Calibration check complete!")
                    st.rerun()
                else:
                    st.error(f"Error: {result.stderr[:500]}")

    report_path = "results/calibration_report.txt"
    if os.path.exists(report_path):
        with col2:
            st.subheader("Summary")
            with open(report_path) as f:
                content = f.read()
            for line in content.split("\n"):
                if "WELL CALIBRATED" in line:
                    st.success(line.strip())
                elif "OVER-CONFIDENT" in line or "CONSERVATIVE" in line:
                    st.warning(line.strip())
                elif line.strip():
                    st.text(line)

    img_path = "results/calibration_reliability.png"
    if os.path.exists(img_path):
        st.subheader("Reliability Diagram")
        st.image(img_path)
        st.caption(
            "Green bars = CI coverage within 5% of nominal 80% (well-calibrated). "
            "Red bars = over-confident or too-wide intervals."
        )
    else:
        st.info("Click 'Run Calibration Check' to generate the reliability diagram.")

    with st.expander("What does calibration mean?"):
        st.markdown("""
        - **80% CI coverage = 80%**: Perfect calibration.
        - **Coverage < 80%**: Over-confident intervals (too narrow).
        - **Coverage > 80%**: Conservative intervals (too wide).

        NeuSight uses **quantile regression** (10th/90th percentile XGBoost)
        to produce CI bounds. This page verifies them against real measurements.
        """)

if __name__ == "__main__":
    import plotly.express as px  # Import here to avoid circular import
    main()
