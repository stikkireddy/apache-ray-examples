# Databricks notebook source
# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## [Batch Training](https://docs.ray.io/en/latest/ray-core/examples/batch_training.html)

# COMMAND ----------

from typing import Callable, Optional, List, Union, Tuple, Iterable
import time
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import pyarrow as pa
from pyarrow import fs
from pyarrow import dataset as ds
from pyarrow import parquet as pq
import pyarrow.compute as pc

PRINT_TIMES = False


def print_time(msg: str):
    if PRINT_TIMES:
        print(msg)

SMOKE_TEST = True

# COMMAND ----------

def read_data(file: str, pickup_location_id: int) -> pd.DataFrame:
    return pq.read_table(
        file,
        filters=[("pickup_location_id", "=", pickup_location_id)],
        columns=[
            "pickup_at",
            "dropoff_at",
            "pickup_location_id",
            "dropoff_location_id",
        ],
    ).to_pandas()

# COMMAND ----------

def transform_batch(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_at"] = pd.to_datetime(
        df["pickup_at"], format="%Y-%m-%d %H:%M:%S"
    )
    df["dropoff_at"] = pd.to_datetime(
        df["dropoff_at"], format="%Y-%m-%d %H:%M:%S"
    )
    df["trip_duration"] = (df["dropoff_at"] - df["pickup_at"]).dt.seconds
    df["pickup_location_id"] = df["pickup_location_id"].fillna(-1)
    df["dropoff_location_id"] = df["dropoff_location_id"].fillna(-1)
    return df

# COMMAND ----------

# Ray task to fit and score a scikit-learn model.
@ray.remote
def fit_and_score_sklearn(
    train: pd.DataFrame, test: pd.DataFrame, model: BaseEstimator
) -> Tuple[BaseEstimator, float]:
    train_X = train[["dropoff_location_id"]]
    train_y = train["trip_duration"]
    test_X = test[["dropoff_location_id"]]
    test_y = test["trip_duration"]

    # Start training.
    model = model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    error = mean_absolute_error(test_y, pred_y)
    return model, error

# COMMAND ----------

def train_and_evaluate_internal(
    df: pd.DataFrame, models: List[BaseEstimator], pickup_location_id: int = 0
) -> List[Tuple[BaseEstimator, float]]:
    # We need at least 4 rows to create a train / test split.
    if len(df) < 4:
        print(
            f"Dataframe for LocID: {pickup_location_id} is empty or smaller than 4"
        )
        return None

    # Train / test split.
    train, test = train_test_split(df)

    # We put the train & test dataframes into Ray object store
    # so that they can be reused by all models fitted here.
    # https://docs.ray.io/en/master/ray-core/patterns/pass-large-arg-by-value.html
    train_ref = ray.put(train)
    test_ref = ray.put(test)

    # Launch a fit and score task for each model.
    results = ray.get(
        [
            fit_and_score_sklearn.remote(train_ref, test_ref, model)
            for model in models
        ]
    )
    results.sort(key=lambda x: x[1])  # sort by error
    return results


@ray.remote
def train_and_evaluate(
    file_name: str,
    pickup_location_id: int,
    models: List[BaseEstimator],
) -> Tuple[str, str, List[Tuple[BaseEstimator, float]]]:
    start_time = time.time()
    data = read_data(file_name, pickup_location_id)
    data_loading_time = time.time() - start_time
    print_time(
        f"Data loading time for LocID: {pickup_location_id}: {data_loading_time}"
    )

    # Perform transformation
    start_time = time.time()
    data = transform_batch(data)
    transform_time = time.time() - start_time
    print_time(
        f"Data transform time for LocID: {pickup_location_id}: {transform_time}"
    )

    # Perform training & evaluation for each model
    start_time = time.time()
    results = (train_and_evaluate_internal(data, models, pickup_location_id),)
    training_time = time.time() - start_time
    print_time(
        f"Training time for LocID: {pickup_location_id}: {training_time}"
    )

    return (
        file_name,
        pickup_location_id,
        results,
    )

# COMMAND ----------

def run_batch_training(files: List[str], models: List[BaseEstimator]):
    print("Starting run...")
    start = time.time()

    # Store task references
    task_refs = []
    for file in files:
        try:
            locdf = pq.read_table(file, columns=["pickup_location_id"])
        except Exception:
            continue
        pickup_location_ids = locdf["pickup_location_id"].unique()

        for pickup_location_id in pickup_location_ids:
            # Cast PyArrow scalar to Python if needed.
            try:
                pickup_location_id = pickup_location_id.as_py()
            except Exception:
                pass
            task_refs.append(
                train_and_evaluate.remote(file, pickup_location_id, models)
            )

    # Block to obtain results from each task
    results = ray.get(task_refs)

    taken = time.time() - start
    count = len(results)
    # If result is None, then it means there weren't enough records to train
    results_not_none = [x for x in results if x is not None]
    count_not_none = len(results_not_none)

    # Sleep a moment for nicer output
    time.sleep(1)
    print("", flush=True)
    print(f"Number of pickup locations: {count}")
    print(
        f"Number of pickup locations with enough records to train: {count_not_none}"
    )
    print(f"Number of models trained: {count_not_none * len(models)}")
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")
    return results

# COMMAND ----------

# Obtain the dataset. Each month is a separate file.
dataset = ds.dataset(
    "s3://anonymous@air-example-data/ursa-labs-taxi-data/by_year/",
    partitioning=["year", "month"],
)
starting_idx = -2 if SMOKE_TEST else 0
files = [f"s3://anonymous@{file}" for file in dataset.files][starting_idx:]
print(f"Obtained {len(files)} files!")

# COMMAND ----------

from sklearn.linear_model import LinearRegression

results = run_batch_training(files, models=[LinearRegression()])
print(results[:10])

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor

results = run_batch_training(
    files,
    models=[
        LinearRegression(),
        DecisionTreeRegressor(),
        DecisionTreeRegressor(splitter="random"),
    ],
)
print(results[:10])

# COMMAND ----------

# Redefine the train_and_evaluate task to use in-memory data.
# We still keep file_name and pickup_location_id for identification purposes.
@ray.remote
def train_and_evaluate(
    pickup_location_id_and_data: Tuple[int, pd.DataFrame],
    file_name: str,
    models: List[BaseEstimator],
) -> Tuple[str, str, List[Tuple[BaseEstimator, float]]]:
    pickup_location_id, data = pickup_location_id_and_data
    # Perform transformation
    start_time = time.time()
    # The underlying numpy arrays are stored in the Ray object
    # store for efficient access, making them immutable. We therefore
    # copy the DataFrame to obtain a mutable copy we can transform.
    data = data.copy()
    data = transform_batch(data)
    transform_time = time.time() - start_time
    print_time(
        f"Data transform time for LocID: {pickup_location_id}: {transform_time}"
    )

    return (
        file_name,
        pickup_location_id,
        train_and_evaluate_internal(data, models, pickup_location_id),
    )


# This allows us to create a Ray Task that is also a generator, returning object references.
@ray.remote(num_returns="dynamic")
def read_into_object_store(file: str) -> ray.ObjectRefGenerator:
    print(f"Loading {file}")
    # Read the entire file into memory.
    try:
        locdf = pq.read_table(
            file,
            columns=[
                "pickup_at",
                "dropoff_at",
                "pickup_location_id",
                "dropoff_location_id",
            ],
        )
    except Exception:
        return []

    pickup_location_ids = locdf["pickup_location_id"].unique()

    for pickup_location_id in pickup_location_ids:
        # Each id-data batch tuple will be put as a separate object into the Ray object store.

        # Cast PyArrow scalar to Python if needed.
        try:
            pickup_location_id = pickup_location_id.as_py()
        except Exception:
            pass

        yield (
            pickup_location_id,
            locdf.filter(
                pc.equal(locdf["pickup_location_id"], pickup_location_id)
            ).to_pandas(),
        )


def run_batch_training_with_object_store(
    files: List[str], models: List[BaseEstimator]
):
    print("Starting run...")
    start = time.time()

    # Store task references
    task_refs = []

    # Use a SPREAD scheduling strategy to load each
    # file on a separate node as an OOM safeguard.
    # This is not foolproof though! We can also specify a resource
    # requirement for memory, if we know what is the maximum
    # memory requirement for a single file.
    read_into_object_store_spread = read_into_object_store.options(
        scheduling_strategy="SPREAD"
    )

    # Dictionary of references to read tasks with file names as keys
    read_tasks_by_file = {
        files[file_id]: read_into_object_store_spread.remote(file)
        for file_id, file in enumerate(files)
    }

    for file, read_task_ref in read_tasks_by_file.items():
        # We iterate over references and pass them to the tasks directly
        for pickup_location_id_and_data_batch_ref in iter(ray.get(read_task_ref)):
            task_refs.append(
                train_and_evaluate.remote(
                    pickup_location_id_and_data_batch_ref, file, models
                )
            )

    # Block to obtain results from each task
    results = ray.get(task_refs)

    taken = time.time() - start
    count = len(results)
    # If result is None, then it means there weren't enough records to train
    results_not_none = [x for x in results if x is not None]
    count_not_none = len(results_not_none)

    # Sleep a moment for nicer output
    time.sleep(1)
    print("", flush=True)
    print(f"Number of pickup locations: {count}")
    print(
        f"Number of pickup locations with enough records to train: {count_not_none}"
    )
    print(f"Number of models trained: {count_not_none * len(models)}")
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")
    return results

# COMMAND ----------

results = run_batch_training_with_object_store(
    files, models=[LinearRegression()]
)
print(results[:10])

# COMMAND ----------

from ray.util.spark import shutdown_ray_cluster
shutdown_ray_cluster()
