# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Install Apache Ray (requires runtime >=12.0)

# COMMAND ----------

# MAGIC %pip install "ray[default] @ https://ml-team-public-read.s3.us-west-2.amazonaws.com/ray-pkgs/demo0113/ray-3.0.0.dev0-cp39-cp39-linux_x86_64.whl"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # setup_ray_cluster options
# MAGIC 
# MAGIC 
# MAGIC * **num_worker_nodes**: This argument represents how many ray worker nodes to start
# MAGIC             for the ray cluster.
# MAGIC             Specifying the `num_worker_nodes` as `ray.util.spark.MAX_NUM_WORKER_NODES`
# MAGIC             represents a ray cluster
# MAGIC             configuration that will use all available resources configured for the
# MAGIC             spark application.
# MAGIC             To create a spark application that is intended to exclusively run a
# MAGIC             shared ray cluster, it is recommended to set this argument to
# MAGIC             **`ray.util.spark.MAX_NUM_WORKER_NODES`**.
# MAGIC * **num_cpus_per_node**: Number of cpus available to per-ray worker node, if not
# MAGIC             provided, use spark application configuration 'spark.task.cpus' instead.
# MAGIC             Limitation: Only spark version >= 3.4 or Databricks Runtime 12.x supports
# MAGIC             setting this argument.
# MAGIC * **num_gpus_per_node**: Number of gpus available to per-ray worker node, if not
# MAGIC             provided, use spark application configuration
# MAGIC             'spark.task.resource.gpu.amount' instead.
# MAGIC             This argument is only available on spark cluster that is configured with
# MAGIC             'gpu' resources.
# MAGIC             Limitation: Only spark version >= 3.4 or Databricks Runtime 12.x supports
# MAGIC             setting this argument.
# MAGIC * **object_store_memory_per_node**: Object store memory available to per-ray worker
# MAGIC             node, but it is capped by
# MAGIC             "dev_shm_available_size * 0.8 / num_tasks_per_spark_worker".
# MAGIC             The default value equals to
# MAGIC             "0.3 * spark_worker_physical_memory * 0.8 / num_tasks_per_spark_worker".
# MAGIC * **head_node_options**: A dict representing Ray head node extra options.
# MAGIC * **worker_node_options**: A dict representing Ray worker node extra options.
# MAGIC * **ray_temp_root_dir**: A local disk path to store the ray temporary data. The
# MAGIC             created cluster will create a subdirectory
# MAGIC             "ray-{head_port}-{random_suffix}" beneath this path.
# MAGIC * **safe_mode**: Boolean flag to fast-fail initialization of the ray cluster if
# MAGIC             the available spark cluster does not have sufficient resources to fulfill
# MAGIC             the resource allocation for memory, cpu and gpu. When set to true, if the
# MAGIC             requested resources are not available for minimum recommended
# MAGIC             functionality, an exception will be raised that details the inadequate
# MAGIC             spark cluster configuration settings. If overridden as `False`,
# MAGIC             a warning is raised.
# MAGIC * **collect_log_to_path**: If specified, after ray head / worker nodes terminated,
# MAGIC             collect their logs to the specified path. On Databricks Runtime, we
# MAGIC             recommend you to specify a local path starts with '/dbfs/', because the
# MAGIC             path mounts with a centralized storage device and stored data is persisted
# MAGIC             after databricks spark cluster terminated.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## [Monte Carlo Example](https://docs.ray.io/en/latest/ray-core/examples/monte_carlo_pi.html)

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
setup_ray_cluster(
  num_worker_nodes=2,
  num_cpus_per_node=1,
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Initialize connection with ray cluster

# COMMAND ----------

import ray
ray.init()

# COMMAND ----------

import math
import time
import random

# COMMAND ----------

@ray.remote
class ProgressActor:
    def __init__(self, total_num_samples: int):
        self.total_num_samples = total_num_samples
        self.num_samples_completed_per_task = {}

    def report_progress(self, task_id: int, num_samples_completed: int) -> None:
        self.num_samples_completed_per_task[task_id] = num_samples_completed

    def get_progress(self) -> float:
        return (
            sum(self.num_samples_completed_per_task.values()) / self.total_num_samples
        )

@ray.remote
def sampling_task(num_samples: int, task_id: int,
                  progress_actor: ray.actor.ActorHandle) -> int:
    num_inside = 0
    for i in range(num_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if math.hypot(x, y) <= 1:
            num_inside += 1

        # Report progress every 1 million samples.
        if (i + 1) % 1_000_000 == 0:
            # This is async.
            progress_actor.report_progress.remote(task_id, i + 1)

    # Report the final progress.
    progress_actor.report_progress.remote(task_id, num_samples)
    return num_inside

# COMMAND ----------

# Change this to match your cluster scale.
NUM_SAMPLING_TASKS = 10
NUM_SAMPLES_PER_TASK = 10_000_000
TOTAL_NUM_SAMPLES = NUM_SAMPLING_TASKS * NUM_SAMPLES_PER_TASK

# Create the progress actor.
progress_actor = ProgressActor.remote(TOTAL_NUM_SAMPLES)

# Create and execute all sampling tasks in parallel.
results = [
    sampling_task.remote(NUM_SAMPLES_PER_TASK, i, progress_actor)
    for i in range(NUM_SAMPLING_TASKS)
]

# Query progress periodically.
while True:
    progress = ray.get(progress_actor.get_progress.remote())
    print(f"Progress: {int(progress * 100)}%")

    if progress == 1:
        break

    time.sleep(1)


# Get all the sampling tasks results.
total_num_inside = sum(ray.get(results))
pi = (total_num_inside * 4) / TOTAL_NUM_SAMPLES
print(f"Estimated value of π is: {pi}")

# COMMAND ----------

shutdown_ray_cluster()
