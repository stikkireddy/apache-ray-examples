# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Install Apache Ray (requires runtime >=12.0)

# COMMAND ----------

# MAGIC %pip install "ray[default] @ https://ml-team-public-read.s3.us-west-2.amazonaws.com/ray-pkgs/demo0113/ray-3.0.0.dev0-cp39-cp39-linux_x86_64.whl"

# COMMAND ----------

# MAGIC %pip install statsforecast

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
# MAGIC ## [Simple AutoML for time series with Ray Core](https://docs.ray.io/en/latest/ray-core/examples/automl_for_time_series.html)

# COMMAND ----------

from typing import List, Union, Callable, Dict, Type, Tuple
import time
import itertools
import pandas as pd
import numpy as np
from collections import defaultdict
from statsforecast import StatsForecast
from statsforecast.models import ETS, AutoARIMA, _TS
from pyarrow import parquet as pq
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# COMMAND ----------

@ray.remote
def train_and_evaluate_fold(
    model: _TS,
    df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    label_column: str,
    metrics: Dict[str, Callable[[pd.Series, pd.Series], float]],
    freq: str = "D",
) -> Dict[str, float]:
    try:
        # Create the StatsForecast object with train data & model.
        statsforecast = StatsForecast(
            df=df.iloc[train_indices], models=[model], freq=freq
        )
        # Make a forecast and calculate metrics on test data.
        # This will fit the model first automatically.
        forecast = statsforecast.forecast(len(test_indices))
        return {
            metric_name: metric(
                df.iloc[test_indices][label_column], forecast[model.__class__.__name__]
            )
            for metric_name, metric in metrics.items()
        }
    except Exception:
        # In case the model fit or eval fails, return None for all metrics.
        return {metric_name: None for metric_name, metric in metrics.items()}

# COMMAND ----------

def evaluate_models_with_cv(
    models: List[_TS],
    df: pd.DataFrame,
    label_column: str,
    metrics: Dict[str, Callable[[pd.Series, pd.Series], float]],
    freq: str = "D",
    cv: Union[int, TimeSeriesSplit] = 5,
) -> Dict[_TS, Dict[str, float]]:
    # Obtain CV train-test indices for each fold.
    if isinstance(cv, int):
        cv = TimeSeriesSplit(cv)
    train_test_indices = list(cv.split(df))

    # Put df into Ray object store for better performance.
    df_ref = ray.put(df)

    # Add tasks to be executed for each fold.
    fold_refs = []
    for model in models:
        fold_refs.extend(
            [
                train_and_evaluate_fold.remote(
                    model,
                    df_ref,
                    train_indices,
                    test_indices,
                    label_column,
                    metrics,
                    freq=freq,
                )
                for train_indices, test_indices in train_test_indices
            ]
        )

    fold_results = ray.get(fold_refs)

    # Split fold results into a list of CV splits-sized chunks.
    # Ray guarantees that order is preserved.
    fold_results_per_model = [
        fold_results[i : i + len(train_test_indices)]
        for i in range(0, len(fold_results), len(train_test_indices))
    ]

    # Aggregate and average results from all folds per model.
    # We go from a list of dicts to a dict of lists and then
    # get a mean of those lists.
    mean_results_per_model = []
    for model_results in fold_results_per_model:
        aggregated_results = defaultdict(list)
        for fold_result in model_results:
            for metric, value in fold_result.items():
                aggregated_results[metric].append(value)
        mean_results = {
            metric: np.mean(values) for metric, values in aggregated_results.items()
        }
        mean_results_per_model.append(mean_results)

    # Join models and their metrics together.
    mean_results_per_model = {
        models[i]: mean_results_per_model[i] for i in range(len(mean_results_per_model))
    }
    return mean_results_per_model

# COMMAND ----------

def generate_configurations(search_space: Dict[Type[_TS], Dict[str, list]]) -> _TS:
    # Convert dict search space into configurations - models instantiated with specific arguments.
    for model, model_search_space in search_space.items():
        kwargs, values = model_search_space.keys(), model_search_space.values()
        # Get a product - all combinations in the per-model grid.
        for configuration in itertools.product(*values):
            yield model(**dict(zip(kwargs, configuration)))


def evaluate_search_space_with_cv(
    search_space: Dict[Type[_TS], Dict[str, list]],
    df: pd.DataFrame,
    label_column: str,
    metrics: Dict[str, Callable[[pd.Series, pd.Series], float]],
    eval_metric: str,
    mode: str = "min",
    freq: str = "D",
    cv: Union[int, TimeSeriesSplit] = 5,
) -> List[Tuple[_TS, Dict[str, float]]]:
    assert eval_metric in metrics
    assert mode in ("min", "max")

    configurations = list(generate_configurations(search_space))
    print(
        f"Evaluating {len(configurations)} configurations with {cv.get_n_splits()} splits each, "
        f"totalling {len(configurations)*cv.get_n_splits()} tasks..."
    )
    ret = evaluate_models_with_cv(
        configurations, df, label_column, metrics, freq=freq, cv=cv
    )

    # Sort the results by eval_metric
    ret = sorted(ret.items(), key=lambda x: x[1][eval_metric], reverse=(mode == "max"))
    print("Evaluation complete!")
    return ret

# COMMAND ----------

def get_m5_partition(unique_id: str) -> pd.DataFrame:
    ds1 = pq.read_table(
        "s3://anonymous@m5-benchmarks/data/train/target.parquet",
        filters=[("item_id", "=", unique_id)],
    )
    Y_df = ds1.to_pandas()
    # StatsForecasts expects specific column names!
    Y_df = Y_df.rename(
        columns={"item_id": "unique_id", "timestamp": "ds", "demand": "y"}
    )
    Y_df["unique_id"] = Y_df["unique_id"].astype(str)
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])
    Y_df = Y_df.dropna()
    constant = 10
    Y_df["y"] += constant
    return Y_df[Y_df.unique_id == unique_id]

# COMMAND ----------

df = get_m5_partition("FOODS_1_001_CA_1")
df

# COMMAND ----------

tuning_results = evaluate_search_space_with_cv(
    {AutoARIMA: {}, ETS: {"season_length": [6, 7], "model": ["ZNA", "ZZZ"]}},
    df,
    "y",
    {"mse": mean_squared_error, "mae": mean_absolute_error},
    "mse",
    cv=TimeSeriesSplit(test_size=1),
)


# COMMAND ----------

print(tuning_results[0])

# Print arguments of the model:
print(tuning_results[0][0].__dict__)

# COMMAND ----------

shutdown_ray_cluster()

# COMMAND ----------


