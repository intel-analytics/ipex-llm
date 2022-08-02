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

3. **create `run.sh` under the subdirectory `<workload>`**

    The `run.sh` is used to run workload, the environment variables which can be used in it directly are:

    - `CONDA`: the install directory of conda
    - `ANALYTICS_ZOO_ROOT`: the root directory of BigDL repo
    - `$ANALYTICS_ZOO_ROOT/python/nano/benchmark`: this directory

    The commands which can be used in it directly are:

    - `git`
    - `wget`
    - `bash`
    - `$CONDA/bin/conda`
    - `source $CONDA/bin/activate <env_name>`: activate a conda environment
    - `python`, `pip`: only can be used after activating conda environment

    You can use `apt-get` to install what you want.

    Use this `run.sh` to install required environment and run your workload

**note:**

- see `resnet50` workload for a reference
