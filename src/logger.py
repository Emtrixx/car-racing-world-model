
import mlflow
import os

class ExperimentLogger:
    def __init__(self, log_dir="logs", experiment_name="default_experiment"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        mlflow.set_tracking_uri(f"file://{os.path.abspath(self.log_dir)}")
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name=None, config=None):
        mlflow.start_run(run_name=run_name)
        if config:
            self.log_params(config)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metric(self, key, value, step):
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics, step):
        mlflow.log_metrics(metrics, step=step)

    def end_run(self):
        mlflow.end_run()
