from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import os
import json
from prediction_api import load_model, predict_for_custom_model, create_sample_model

app = Flask(__name__)

# Load the prediction model
prediction_model = load_model('models/gradient_boosting_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract model parameters
    num_layers = int(data.get('num_layers', 3))
    channels = int(data.get('channels', 16))
    batch_sizes = [int(bs) for bs in data.get('batch_sizes', [1, 2, 4, 8])]
    
    # Create a sample model with the specified parameters
    model = create_sample_model(num_layers, channels)
    
    # Make predictions
    predictions = predict_for_custom_model(
        prediction_model, 
        model, 
        (3, 224, 224), 
        batch_sizes
    )
    
    return jsonify(predictions)

@app.route('/models')
def list_models():
    # List available pre-defined models
    models = [
        {"name": "Simple CNN (3 layers)", "params": {"num_layers": 3, "channels": 16}},
        {"name": "Simple CNN (5 layers)", "params": {"num_layers": 5, "channels": 16}},
        {"name": "Wide CNN (3 layers)", "params": {"num_layers": 3, "channels": 32}}
    ]
    return jsonify(models)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create a simple HTML template
    with open('templates/index.html', 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>GPU Usage Prediction</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 800px; margin: 0 auto; }
                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; }
                input, select { width: 100%; padding: 8px; box-sizing: border-box; }
                button { background: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
                .results { margin-top: 20px; border: 1px solid #ddd; padding: 15px; }
                table { width: 100%; border-collapse: collapse; }
                table, th, td { border: 1px solid #ddd; }
                th, td { padding: 8px; text-align: left; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>GPU Usage Prediction</h1>
                <div class="form-group">
                    <label for="modelType">Model Type:</label>
                    <select id="modelType">
                        <option value="custom">Custom Model</option>
                    </select>
                </div>
                <div id="customParams">
                    <div class="form-group">
                        <label for="numLayers">Number of Layers:</label>
                        <input type="number" id="numLayers" value="3" min="1" max="10">
                    </div>
                    <div class="form-group">
                        <label for="channels">Base Channels:</label>
                        <input type="number" id="channels" value="16" min="8" max="128">
                    </div>
                </div>
                <div class="form-group">
                    <label for="batchSizes">Batch Sizes (comma-separated):</label>
                    <input type="text" id="batchSizes" value="1,2,4,8">
                </div>
                <button onclick="predict()">Predict GPU Usage</button>
                
                <div id="results" class="results" style="display: none;">
                    <h2>Prediction Results</h2>
                    <div id="modelInfo"></div>
                    <table id="predictionsTable">
                        <thead>
                            <tr>
                                <th>Batch Size</th>
                                <th>Execution Time (ms)</th>
                                <th>Memory Usage (MB)</th>
                            </tr>
                        </thead>
                        <tbody id="predictionsBody">
                        </tbody>
                    </table>
                </div>
            </div>
            
            <script>
                // Load available models
                fetch('/models')
                    .then(response => response.json())
                    .then(models => {
                        const select = document.getElementById('modelType');
                        models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = JSON.stringify(model.params);
                            option.textContent = model.name;
                            select.appendChild(option);
                        });
                    });
                
                // Handle model type change
                document.getElementById('modelType').addEventListener('change', function() {
                    const value = this.value;
                    if (value === 'custom') {
                        document.getElementById('customParams').style.display = 'block';
                    } else {
                        document.getElementById('customParams').style.display = 'none';
                        try {
                            const params = JSON.parse(value);
                            document.getElementById('numLayers').value = params.num_layers;
                            document.getElementById('channels').value = params.channels;
                        } catch (e) {
                            console.error('Invalid JSON:', e);
                        }
                    }
                });
                
                function predict() {
                    const numLayers = parseInt(document.getElementById('numLayers').value);
                    const channels = parseInt(document.getElementById('channels').value);
                    const batchSizes = document.getElementById('batchSizes').value
                        .split(',')
                        .map(s => parseInt(s.trim()))
                        .filter(n => !isNaN(n));
                    
                    const data = {
                        num_layers: numLayers,
                        channels: channels,
                        batch_sizes: batchSizes
                    };
                    
                    fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(data)
                    })
                    .then(response => response.json())
                    .then(predictions => {
                        document.getElementById('results').style.display = 'block';
                        
                        // Display model info
                        const modelInfo = document.getElementById('modelInfo');
                        modelInfo.innerHTML = `
                            <p><strong>Model Type:</strong> Custom CNN</p>
                            <p><strong>Layers:</strong> ${numLayers}</p>
                            <p><strong>Base Channels:</strong> ${channels}</p>
                        `;
                        
                        // Display predictions
                        const tbody = document.getElementById('predictionsBody');
                        tbody.innerHTML = '';
                        
                        predictions.forEach(pred => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td>${pred.batch_size}</td>
                                <td>${pred.predicted_execution_time_ms.toFixed(2)}</td>
                                <td>${pred.predicted_memory_usage_mb.toFixed(2)}</td>
                            `;
                            tbody.appendChild(row);
                        });
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('An error occurred while making predictions.');
                    });
                }
            </script>
        </body>
        </html>
        ''')
    
    app.run(debug=True, port=5000)
