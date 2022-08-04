# BigDL Nano Benchmark (in progress)

This directory contains workloads used for bigdl-nano preformance regression test, this document shows how to add new workloads:

## Directory Structure

Every subdirectory under the current directory is a workload, the name of subdirectory is the name of workload.

Every subdirectory should contain a `run.sh`, which is used for running workload.

No other rules to follow.

For example, the `resnet50` subdirectory means a workload named `resnet50`, and the `run.sh` under it will be used to run this workload.

## Add New Workload

1. **create a subdirectory `<workload>` under current directory**

2. **implement your workload code in `<workload>` directory**

    Your workload code should print benchmark result to the standard output,
    and use '>>>' and '<<<' to enclose it.

    The benchmark result itself should be a JSON string, which contains a field named 'config' meaning the configuration of this test, you can define any other fields to store the metrics which you are intrested in.

    For example
    ```python
    output = json.dumps({
        "config": "Running Nano default with ipex 4 processes",
        "train time": train_time,
        "other metrics": other_metrics,
    })
    print(f'>>>{output}<<<')
    ```

3. **create `run.sh` under the subdirectory `<workload>`**

    The `run.sh` is used to run workload, the environment variables which can be used in it directly are:

    - `ANALYTICS_ZOO_ROOT`: the root directory of BigDL repo
    - `$ANALYTICS_ZOO_ROOT/python/nano/benchmark`: this directory

    Use this `run.sh` to install required environment and run your workload

**note:**

- see `resnet50` workload for a reference
