class PerformanceMonitor:
    def __init__(self, prediction_model, error_threshold=20):
        self.prediction_model = prediction_model
        self.error_threshold = error_threshold
        self.performance_history = {}
        
    def record_performance(self, model_name, batch_size, predicted_time, actual_time):
        """Record performance and check for anomalies"""
        key = f"{model_name}_{batch_size}"
        
        # Calculate error
        error_percent = abs(predicted_time - actual_time) / actual_time * 100
        
        # Record in history
        if key not in self.performance_history:
            self.performance_history[key] = []
            
        self.performance_history[key].append({
            'timestamp': datetime.now(),
            'predicted_time': predicted_time,
            'actual_time': actual_time,
            'error_percent': error_percent
        })
        
        # Check for anomaly
        is_anomaly = error_percent > self.error_threshold
        
        # Log feedback for model improvement
        if is_anomaly:
            print(f"Performance anomaly detected for {model_name} with batch size {batch_size}")
            print(f"Predicted: {predicted_time:.2f} ms, Actual: {actual_time:.2f} ms, Error: {error_percent:.2f}%")
            
            # Add feedback to prediction model
            features = {
                'model_name': model_name,
                'batch_size': batch_size,
                # Add other required features
            }
            self.prediction_model.log_feedback(features, predicted_time, actual_time)
            
        return is_anomaly
    
    def get_performance_trend(self, model_name, batch_size, window=10):
        """Get recent performance trend for a specific model and batch size"""
        key = f"{model_name}_{batch_size}"
        
        if key not in self.performance_history or len(self.performance_history[key]) < 2:
            return None
            
        # Get recent entries
        recent = self.performance_history[key][-window:]
        
        # Calculate trend
        errors = [entry['error_percent'] for entry in recent]
        avg_error = sum(errors) / len(errors)
        trend = errors[-1] - errors[0]  # Positive means error is increasing
        
        return {
            'average_error': avg_error,
            'trend': trend,
            'is_improving': trend < 0
        }
