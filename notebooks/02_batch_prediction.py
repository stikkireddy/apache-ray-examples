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

# MAGIC %md
# MAGIC 
# MAGIC ## [Batch Prediction](https://docs.ray.io/en/latest/ray-core/examples/batch_prediction.html)

# COMMAND ----------

import pandas as pd
import numpy as np


def load_model():
    # A dummy model.
    def model(batch: pd.DataFrame) -> pd.DataFrame:
        # Dummy payload so copying the model will actually copy some data
        # across nodes.
        model.payload = np.zeros(100_000_000)
        return pd.DataFrame({"score": batch["passenger_count"] % 2 == 0})

    return model

# COMMAND ----------

import pyarrow.parquet as pq
import ray

@ray.remote
def make_prediction(model, shard_path):
    df = pq.read_table(shard_path).to_pandas()
    result = model(df)

    # Write out the prediction result.
    # NOTE: unless the driver will have to further process the
    # result (other than simply writing out to storage system),
    # writing out at remote task is recommended, as it can avoid
    # congesting or overloading the driver.
    # ...

    # Here we just return the size about the result in this example.
    return len(result)

# COMMAND ----------

# 12 files, one for each remote task.
input_files = [
        f"s3://anonymous@air-example-data/ursa-labs-taxi-data/downsampled_2009_full_year_data.parquet"
        f"/fe41422b01c04169af2a65a83b753e0f_{i:06d}.parquet"
        for i in range(12)
]

# ray.put() the model just once to local object store, and then pass the
# reference to the remote tasks.
model = load_model()
model_ref = ray.put(model)

result_refs = []

# Launch all prediction tasks.
for file in input_files:
    # Launch a prediction task by passing model reference and shard file to it.
    # NOTE: it would be highly inefficient if you are passing the model itself
    # like make_prediction.remote(model, file), which in order to pass the model
    # to remote node will ray.put(model) for each task, potentially overwhelming
    # the local object store and causing out-of-disk error.
    result_refs.append(make_prediction.remote(model_ref, file))

results = ray.get(result_refs)

# Let's check prediction output size.
for r in results:
    print("Prediction output size:", r)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Batch Prediction with Actors

# COMMAND ----------

import pandas as pd
import pyarrow.parquet as pq
import ray

@ray.remote
class BatchPredictor:
    def __init__(self, model):
        self.model = model
        
    def predict(self, shard_path):
        df = pq.read_table(shard_path).to_pandas()
        result =self.model(df)

        # Write out the prediction result.
        # NOTE: unless the driver will have to further process the
        # result (other than simply writing out to storage system),
        # writing out at remote task is recommended, as it can avoid
        # congesting or overloading the driver.
        # ...

        # Here we just return the size about the result in this example.
        return len(result)

# COMMAND ----------

from ray.util.actor_pool import ActorPool

model = load_model()
model_ref = ray.put(model)
num_actors = 4
actors = [BatchPredictor.remote(model_ref) for _ in range(num_actors)]
pool = ActorPool(actors)
input_files = [
        f"s3://anonymous@air-example-data/ursa-labs-taxi-data/downsampled_2009_full_year_data.parquet"
        f"/fe41422b01c04169af2a65a83b753e0f_{i:06d}.parquet"
        for i in range(12)
]
for file in input_files:
    pool.submit(lambda a, v: a.predict.remote(v), file)
while pool.has_next():
    print("Prediction output size:", pool.get_next())

# COMMAND ----------

shutdown_ray_cluster()

# COMMAND ----------


