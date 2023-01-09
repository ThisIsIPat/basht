import os
import random
from os import path
from time import sleep
from typing import Generator

import numpy
import ray
from kubernetes import client, config, watch
from kubernetes.client import ApiException
from kubernetes.utils import FailToCreateError
from ray import tune

from ml_benchmark.benchmark_runner import Benchmark
from ml_benchmark.workload.mnist.mnist_task import MnistTask
from ml_benchmark.utils.yaml import YMLHandler


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
        self.namespace = resources.get("kubernetesNamespace", f"raytune-study-{random.randint(0, 10024)}")
        self.master_ip = resources.get("kubernetesMasterIP")
        self.trial_tag = resources.get("dockerImageTag", "raytune-trial:latest")
        self.study_name = resources.get("studyName", "raytune-study")
        self.workerCpu = resources.get("workerCpu", 2)
        self.workerMemory = resources.get("workerMemory", 2)
        self.workerCount = resources.get("workerCount", 4)
        self.delete_after_run = resources.get("deleteAfterRun", True)
        self.metrics_ip = resources.get("metricsIP")
        self.trials = resources.get("trials", 10)
        self.epochs = resources.get("epochs", 5)
        self.hyperparameter = resources.get("hyperparameter")

    def deploy(self) -> None:
        """
        Deploy Raytune Operator and Head.
        """

        # generate hyperparameter file from resources def.
        if self.hyperparameter:
            f = path.join(path.dirname(__file__), "hyperparameter_space.yml")
            YMLHandler.as_yaml(f, self.hyperparameter)

        try:
            resp = client.CoreV1Api().create_namespace(
                client.V1Namespace(metadata=client.V1ObjectMeta(name=self.namespace)))
            print("Namespace created. status='%s'" % str(resp))
        except ApiException as e:
            if self._is_create_conflict(e):
                print("Deployment already exists")
            else:
                raise e

        # TODO: Possibly conflicting ray-system namespace on parallel runs? Is this problematic?
        # Solution is likely to accept the error - shouldn't be a problem to have multiple users on the same operator
        # Create Ray Operator
        ray_operator_kustomize_config_path = path.join(
            path.dirname(__file__), "ops/manifests/operator/config/default")
        ray_cluster_kustomize_config_template_path = path.join(
            path.dirname(__file__), "ops/manifests/ray_resources_template.yaml")
        ray_cluster_kustomize_config_path = path.join(
            path.dirname(__file__), "ops/manifests/ray_resources.yaml")

        ray_template = YMLHandler.load_yaml(ray_cluster_kustomize_config_template_path)
        # Assuming that the worker CPU and memory are set to a value occupying a single node each,
        # the head should be set to the same value to ensure that the head has its own node.

        # Slightly overprovision CPU. Ray puts the limit of 2 CPU explicitly anyways
        ray_template["spec"]["headGroupSpec"]["template"]["spec"]["containers"][0]["resources"] = {
            "requests": {
                "cpu": f"{self.workerCpu}048m",
                "memory": f"{self.workerMemory}Gi"
            },
            "limits": {
                "cpu": f"{self.workerCpu}048m",
                "memory": f"{self.workerMemory}Gi"
            }
        }
        ray_template["spec"]["workerGroupSpecs"][0]["template"]["spec"]["containers"][0]["resources"] = {
            "requests": {
                "cpu": f"{self.workerCpu}048m",
                "memory": f"{self.workerMemory}Gi"
            },
            "limits": {
                "cpu": f"{self.workerCpu}048m",
                "memory": f"{self.workerMemory}Gi"
            }
        }
        ray_template["spec"]["workerGroupSpecs"][0]["replicas"] = self.workerCount
        ray_template["spec"]["workerGroupSpecs"][0]["minReplicas"] = self.workerCount
        ray_template["spec"]["workerGroupSpecs"][0]["maxReplicas"] = self.workerCount
        YMLHandler.as_yaml(ray_cluster_kustomize_config_path, ray_template)

        print("Creating Ray Operator...")
        # Running server-side patches is not supported in combination with kustomize directories:
        # See https://github.com/kubernetes-client/python/issues/1555 and nested issues
        print(subprocess.check_output(
            "kubectl apply --context " + resources.get("kubernetesContext") +
            " --server-side -k " + ray_operator_kustomize_config_path, shell=True
        ).decode("utf-8").strip("\n"))
        print("Ray Operator created.")

        kubernetes_watch = watch.Watch()
        kubernetes_watch_stream = \
            kubernetes_watch.stream(client.CoreV1Api().list_namespaced_pod, namespace=self.namespace)

        print("Creating Ray Cluster...")
        # create_from_yaml does not support custom resource definitions due to its reflective API wrapper fetch nature
        print(subprocess.check_output(
            "kubectl apply -n " + self.namespace + " --context " + resources.get("kubernetesContext") +
            " --server-side -f " + ray_cluster_kustomize_config_path, shell=True
        ).decode("utf-8").strip("\n"))
        print("Ray Cluster created.")

        print("Waiting for Ray Head to be ready... (This may take a while.)")
        head_pod_name = self.wait_for_head(kubernetes_watch_stream)
        print("Ray Head is ready.")
        kubernetes_watch.stop()

        # Buffer time for Kubernetes shenanigans, otherwise the port-forwarding will fail even on running state
        # TODO: Better way to detect whether port-forward is possible?
        sleep(4)
        # Process should automatically die when resources are deleted, but manual cleanup is done as well, just in case
        self.port_forward_process = subprocess.Popen(
            "kubectl port-forward -n " + self.namespace + " --context " + resources.get("kubernetesContext") +
            " svc/raycluster-study-head-svc 6379:6379 8265:8265 10001:10001"
        )

        print("Port-forwarding successful. Deployment complete.")

    def wait_for_head(self, kubernetes_watch_stream: Generator) -> str:
        """
        Wait for the head pod to be ready (TODO: And workers!)
        and returns the name of the head pod.
        """
        for event in kubernetes_watch_stream:
            if not event["object"].metadata.name.startswith("raycluster-study-head-"):
                continue
            print("Head now in state \"" + event["object"].status.phase + "\"")
            if event["object"].status.phase == "Running":
                return event["object"].metadata.name
        raise Exception("Kubernetes Watch API stopped functioning")

    @staticmethod
    def _is_create_conflict(e):
        if isinstance(e, ApiException):
            if e.status == 409:
                return True
        if isinstance(e, FailToCreateError):
            if e.api_exceptions is not None:
                # lets quickly check if all status codes are 409 -> componetnes exist already
                if set(map(lambda x: x.status, e.api_exceptions)) == {409}:
                    return True
        return False

    def setup(self):
        """
        Connect to the Ray cluster by calling ray.init
        """
        # Consider replacing master ip with localhost during test runs
        ray.init(address=self.master_ip, runtime_env={
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
            # From simple raytune example
            "hyperparameters": {
                "learning_rate": tune.grid_search(numpy.linspace(0.0001, 0.01, 10)),
                "weight_decay": tune.grid_search(numpy.linspace(0.00001, 0.001, 10)),
                "hidden_layer_config": tune.grid_search([[20], [10, 10]]),
            }
        }

        tuner = tune.Tuner(
            tune.with_resources(ray_objective, {
                # Automatically provisions exactly one worker per trial
                "cpu": self.workerCpu
            }),
            param_space=search_space
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
        print("Stopping port-forward...")
        self.port_forward_process.kill()
        if self.delete_after_run:
            print("Deleting Study namespace...")
            client.CoreV1Api().delete_namespace(self.namespace)
            self.delete_ray_resources()
            print("Waiting for namespace deletion...")
            self._watch_namespace()
            self.image_builder.cleanup(self.trial_tag)

    def delete_ray_resources(self):
        print("Deleting Ray operator namespace...")
        client.CoreV1Api().delete_namespace("ray-system")
        print("Deleting Ray CRD...")
        client.ApiextensionsV1Api().delete_custom_resource_definition("rayclusters.ray.io")
        client.ApiextensionsV1Api().delete_custom_resource_definition("rayservices.ray.io")
        client.ApiextensionsV1Api().delete_custom_resource_definition("rayjobs.ray.io")
        print("Deleting Ray roles...")
        client.RbacAuthorizationV1Api().delete_cluster_role("kuberay-operator")
        client.RbacAuthorizationV1Api().delete_cluster_role_binding("kuberay-operator")

    def _watch_namespace(self):
        try:
            client.CoreV1Api().read_namespace_status(self.namespace).to_dict()
            sleep(2)
        except client.exceptions.ApiException:
            try:
                client.CoreV1Api().read_namespace_status("ray-system").to_dict()
                sleep(2)
            except client.exceptions.ApiException:
                return


if __name__ == "__main__":
    from ml_benchmark.benchmark_runner import BenchmarkRunner
    import subprocess
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
