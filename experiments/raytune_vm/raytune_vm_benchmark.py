import os
from os import path

import numpy
import ray
from kubernetes import config
from ray import tune
from ray.tune import TuneConfig

from ml_benchmark.benchmark_runner import Benchmark
from ml_benchmark.workload.mnist.mnist_task import MnistTask
from ml_benchmark.utils.yaml import YMLHandler


# TODO: Figure out a way to allow ml_benchmark in the worker nodes
# TODO: Consider ray.put/ray.get before to store task objective
def ray_objective(config):
    task = MnistTask(config_init={"epochs": config["epochs"]})
    objective = task.create_objective()
    objective.set_hyperparameters(config["hyperparameters"])
    objective.train()
    validation_scores = objective.validate()
    return {
        "f1-score": validation_scores["macro avg"]["f1-score"]
    }


if os.name == "nt":
    # hacky workaround on Windows
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


class RaytuneKindBenchmark(Benchmark):

    def __init__(self, resources: dict) -> None:
        """
        Processes the given resources dictionary and creates class variables from it which are used
        in the benchmark.

        Args:
            resources (dict): _description_
        """
        config.load_kube_config(context=resources.get("kubernetesContext"))
        self.master_ip = resources.get("vmMasterIP")
        self.trial_tag = resources.get("dockerImageTag", "raytune-trial:latest")
        self.study_name = resources.get("studyName", "raytune-study")
        self.workerCpu = resources.get("workerCpu", 2)
        self.delete_after_run = resources.get("deleteAfterRun", True)
        self.metrics_ip = resources.get("metricsIP")
        self.trials = resources.get("trials", 10)
        self.epochs = resources.get("epochs", 5)
        self.hyperparameter = resources.get("hyperparameter")

    def deploy(self) -> None:
        """
        Expect Ray Head in VM.
        """

        # generate hyperparameter file from resources def.
        if self.hyperparameter:
            f = path.join(path.dirname(__file__), "hyperparameter_space.yml")
            YMLHandler.as_yaml(f, self.hyperparameter)

    def setup(self):
        """
        Connect to the Ray cluster by calling ray.init
        """
        # Consider replacing master ip with localhost during test runs
        ray.init(address="ray://{}:10001".format(self.master_ip),
                 _redis_password="5241590000000000",
                 runtime_env={
            "py_modules": [path.join(path.dirname(__file__), "../../ml_benchmark")],
            "pip": ["torchvision==0.11.3", "torch==1.10.2",
                    # Copied from ml_benchmark/__init__, TODO consider programmatic extraction
                    "scikit-learn==0.24.2",
                    "tqdm==4.62.3", "SQLAlchemy==1.4.31", "docker==4.4.2",
                    "psycopg2-binary",
                    "prometheus-api-client==0.5.1",
                    "ruamel.yaml==0.17.21"
                    ]
        })

    def run(self):
        """
           Start the ray tune task normally, should be connected to cluster
        """
        search_space = {
            "epochs": self.epochs,
            # Reduced from 100 steps to 10 each during testing. Later, make configurable
            # From simple raytune example
            "hyperparameters": {
                "learning_rate": tune.choice(numpy.linspace(0.0001, 0.01, 10)),
                "weight_decay": tune.choice(numpy.linspace(0.00001, 0.001, 10)),
                "hidden_layer_config": tune.grid_search([[20], [10, 10]]),
            }
        }

        tuner = tune.Tuner(
            tune.with_resources(ray_objective, {
                # TODO: Other resources too, like memory? Not necessary bc of ray_resources.yaml but more consistent
                "cpu": self.workerCpu
            }),
            param_space=search_space,
            tune_config=TuneConfig(
                num_samples=self.trials, # TODO: Why 20 instead of 10?
            )
        )
        self.results = tuner.fit()

    def collect_run_results(self):
        """
           Collect RayTune results
        """
        self.best_trial_config = self.results.get_best_result(metric="f1-score", mode="max").config

    # Why is this here? Looks abstractable?
    def test(self):
        def raytune_trial(trial_config):
            objective = MnistTask(config_init={"epochs": 1}).create_objective()

            # Reset ray
            ray.shutdown()

            objective.set_hyperparameters(trial_config["hyperparameters"])
            # these are the results, that can be used for the hyperparameter search
            objective.train()
            validation_scores = objective.validate()
            return validation_scores["macro avg"]["f1-score"]

        self.scores = raytune_trial(self.best_trial_config)

    def collect_benchmark_metrics(self):
        results = dict(
            test_scores=self.scores
        )
        return results

    def undeploy(self):
        """Kill all containers
        """
        # Ray-operator service, pod, deployment etc. get deleted automatically
        # autoscaler etc are deleted automatically by self.namespace deletion
        pass


if __name__ == "__main__":
    from ml_benchmark.benchmark_runner import BenchmarkRunner
    from urllib.request import urlopen
    import os

    # The basic config for the workload. For testing purposes set epochs to one.
    # For benchmarking take the default value of 100
    # your ressources the optimization should run on
    dir_path = os.path.abspath(os.path.dirname(__file__))
    resources = YMLHandler.load_yaml(os.path.join(dir_path, "resource_definition.yml"))
    to_automate = {
        "metricsIP": urlopen("https://checkip.amazonaws.com").read().decode("utf-8").strip(),
        "prometheus_url": "http://localhost:30041"
    }
    resources.update(to_automate)

    # import an use the runner
    runner = BenchmarkRunner(
        benchmark_cls=RaytuneKindBenchmark, resources=resources)
    runner.run()
