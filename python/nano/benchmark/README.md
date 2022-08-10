# BigDL Nano Benchmark

This directory contains workloads used for bigdl-nano preformance regression test, this document shows how to add new workloads:

## Directory Structure

Every subdirectory under this directory is a workload, the name of subdirectory is the name of workload.

Every subdirectory should contain a `run.sh`, which is used for running workload.

For example, the `resnet50` subdirectory means a workload named `resnet50`, and the `run.sh` under it will be used to run this workload.

## Add New Workload

1. **Create a subdirectory `<workload>` under current directory**

2. **Implement your workload code in the `<workload>` directory**

    Your workload code should **print benchmark result to the standard output,
    and use `'>>>'` and `'<<<'` to enclose it.**

    **The benchmark result itself should be a JSON string, which contains a key named `'config'` meaning the configuration of this benchmark,** you can define any other fields to store the metrics which you are intrested in
    
    **Note that there are some other requirements:**
    - all keys in the JSON string must be valid database column names
    - all values in the JSON string must be numbers, strings or datetimes

    For example
    ```python
    output = json.dumps({
        "config": "Running Nano default with ipex 4 processes",
        "train_time": train_time,
        "other_metric": other_metric,
    })
    print(f'>>>{output}<<<')
    ```

3. **Create `run.sh` under the subdirectory `<workload>`**

    The `run.sh` is used to run workload, the environment variables which can be used in it directly are:

    - `ANALYTICS_ZOO_ROOT`: the root directory of BigDL repo
    - `$ANALYTICS_ZOO_ROOT/python/nano/benchmark`: this directory

    Use this `run.sh` to install required environment and run your workload.

4. **Create a job in the `.github/workflows/performance-regression-test.yml` and create a table in the database for your workload**

    If you have finished the above steps, you can contact me to complete this step.


**note:**

- see `resnet50` workload for a reference
- the performance regression test runs automatically every day, but **you can create a comment `'APRT'` in pull request** to trigger it manually
