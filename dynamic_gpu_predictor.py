# dynamic_gpu_predictor.py
from dynamic_predictor import DynamicPredictor
from batch_size_optimizer import BatchSizeOptimizer
from workload_scheduler import WorkloadScheduler
from performance_monitor import PerformanceMonitor

class DynamicGPUPredictor:
    def __init__(self, model_path='models/gradient_boosting_model.joblib', 
                 memory_limit_mb=8000, num_gpus=1):
        # Initialize components
        self.predictor = DynamicPredictor(model_path)
        self.batch_optimizer = BatchSizeOptimizer(self.predictor, memory_limit_mb)
        self.scheduler = WorkloadScheduler(self.predictor, num_gpus)
        self.monitor = PerformanceMonitor(self.predictor)
        
    def predict_and_optimize(self, model_features):
        """Predict execution time and optimize batch size"""
        # Find optimal batch size
        optimal_batch = self.batch_optimizer.find_optimal_batch_size(model_features)
        
        # Update features with optimal batch size
        optimized_features = model_features.copy()
        optimized_features['batch_size'] = optimal_batch
        
        # Make prediction
        predicted_time = self.predictor.predict(optimized_features)
        
        return {
            'optimal_batch_size': optimal_batch,
            'predicted_execution_time': predicted_time,
            'estimated_memory_usage': self.batch_optimizer.estimate_memory_usage(
                model_features, optimal_batch)
        }
    
    def schedule_model(self, model_features, job_id, priority=1):
        """Schedule a model execution on available GPUs"""
        return self.scheduler.schedule_job(model_features, job_id, priority)
    
    def record_actual_performance(self, model_name, batch_size, predicted_time, actual_time):
        """Record actual performance and detect anomalies"""
        is_anomaly = self.monitor.record_performance(
            model_name, batch_size, predicted_time, actual_time)
        
        if is_anomaly:
            # Rebalance workload if anomaly detected
            self.scheduler.rebalance_workload()
            
        return is_anomaly
    
    def get_performance_insights(self, model_name, batch_size):
        """Get performance insights for a specific model"""
        return self.monitor.get_performance_trend(model_name, batch_size)
