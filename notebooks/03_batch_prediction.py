# Databricks notebook source
# MAGIC %run ./00_setup

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

from ray.util.spark import shutdown_ray_cluster
shutdown_ray_cluster()
