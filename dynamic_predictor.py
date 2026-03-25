import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

class DynamicPredictor:
    def __init__(self, model_path='models/gradient_boosting_model.joblib', 
                 feedback_log_path='data/feedback_log.csv'):
        self.model_path = model_path
        self.feedback_log_path = feedback_log_path
        self.model = joblib.load(model_path)
        self.feedback_data = self._load_feedback_data()
        
    def _load_feedback_data(self):
        """Load feedback data or create empty dataframe if not exists"""
        if os.path.exists(self.feedback_log_path):
            return pd.read_csv(self.feedback_log_path)
        else:
            columns = ['timestamp', 'model_name', 'batch_size', 'total_parameters', 
                      'trainable_parameters', 'model_size_mb', 'predicted_time', 
                      'actual_time', 'error_percent']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.feedback_log_path, index=False)
            return df
    
    def predict(self, features):
        """Make prediction using current model"""
        # Convert features to DataFrame for prediction
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = pd.DataFrame(features)
            
        # Make prediction
        prediction = self.model.predict(features_df)[0]
        return prediction
    
    def log_feedback(self, features, predicted_time, actual_time):
        """Log prediction feedback for model improvement"""
        error_percent = abs(predicted_time - actual_time) / actual_time * 100
        
        # Create feedback entry
        feedback = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': features.get('model_name', 'unknown'),
            'batch_size': features.get('batch_size', 0),
            'total_parameters': features.get('total_parameters', 0),
            'trainable_parameters': features.get('trainable_parameters', 0),
            'model_size_mb': features.get('model_size_mb', 0),
            'predicted_time': predicted_time,
            'actual_time': actual_time,
            'error_percent': error_percent
        }
        
        # Append to feedback data
        self.feedback_data = pd.concat([self.feedback_data, pd.DataFrame([feedback])], 
                                      ignore_index=True)
        self.feedback_data.to_csv(self.feedback_log_path, index=False)
        
        print(f"Logged feedback: Error {error_percent:.2f}%")
        
        # Check if retraining is needed
        if len(self.feedback_data) % 10 == 0:  # Retrain every 10 feedback entries
            self.retrain_model()
    
    def retrain_model(self):
        """Retrain model with feedback data"""
        print("Retraining model with feedback data...")
        
        # Load original training data
        raw_data_dir = 'data/raw'
        all_data = []
        
        for filename in os.listdir(raw_data_dir):
            if filename.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(raw_data_dir, filename))
                    all_data.append(df)
                except:
                    pass
        
        if not all_data:
            print("No training data found. Skipping retraining.")
            return
            
        training_data = pd.concat(all_data, ignore_index=True)
        
        # Prepare features and target
        feature_cols = ['total_parameters', 'trainable_parameters', 'model_size_mb', 'batch_size']
        X = training_data[feature_cols]
        y = training_data['execution_time_ms']
        
        # Add feedback data
        feedback_features = self.feedback_data[feature_cols]
        feedback_target = self.feedback_data['actual_time']
        
        X = pd.concat([X, feedback_features], ignore_index=True)
        y = pd.concat([y, feedback_target], ignore_index=True)
        
        # Train new model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        new_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        new_model.fit(X_train, y_train)
        
        # Save new model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_path = f'models/dynamic_gb_model_{timestamp}.joblib'
        joblib.dump(new_model, new_model_path)
        
        # Update current model
        self.model = new_model
        
        print(f"Model retrained and saved to {new_model_path}")
