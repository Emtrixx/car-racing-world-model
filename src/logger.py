from pathlib import Path

import mlflow


class ExperimentLogger:
    def __init__(self, log_dir="logs", experiment_name="default_experiment"):
        # Create a dedicated 'mlruns' subdirectory for MLflow data
        self.mlflow_log_dir = Path(log_dir) / "wm_mlflow"
        self.experiment_name = experiment_name
        self.mlflow_log_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(self.mlflow_log_dir)
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
