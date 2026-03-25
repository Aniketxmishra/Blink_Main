class WorkloadScheduler:
    def __init__(self, prediction_model, num_gpus=1):
        self.prediction_model = prediction_model
        self.num_gpus = num_gpus
        self.current_jobs = [[] for _ in range(num_gpus)]
        self.gpu_loads = [0] * num_gpus
        
    def schedule_job(self, model_features, job_id, priority=1):
        """Schedule a job on the least loaded GPU"""
        # Predict execution time
        exec_time = self.prediction_model.predict(model_features)
        
        # Find least loaded GPU
        least_loaded_idx = self.gpu_loads.index(min(self.gpu_loads))
        
        # Add job to that GPU
        self.current_jobs[least_loaded_idx].append({
            'job_id': job_id,
            'model_name': model_features.get('model_name', 'unknown'),
            'batch_size': model_features.get('batch_size', 1),
            'predicted_time': exec_time,
            'priority': priority
        })
        
        # Update load
        self.gpu_loads[least_loaded_idx] += exec_time * priority
        
        return {
            'assigned_gpu': least_loaded_idx,
            'estimated_start_time': self.gpu_loads[least_loaded_idx] - exec_time * priority,
            'estimated_completion_time': self.gpu_loads[least_loaded_idx]
        }
    
    def rebalance_workload(self):
        """Rebalance workload across GPUs"""
        # Flatten all jobs
        all_jobs = []
        for gpu_jobs in self.current_jobs:
            all_jobs.extend(gpu_jobs)
            
        # Sort by priority (higher first)
        all_jobs.sort(key=lambda x: x['priority'], reverse=True)
        
        # Reset current state
        self.current_jobs = [[] for _ in range(self.num_gpus)]
        self.gpu_loads = [0] * self.num_gpus
        
        # Reschedule all jobs
        schedule_results = []
        for job in all_jobs:
            model_features = {
                'model_name': job['model_name'],
                'batch_size': job['batch_size'],
                # Add other required features
            }
            result = self.schedule_job(model_features, job['job_id'], job['priority'])
            schedule_results.append(result)
            
        return schedule_results
