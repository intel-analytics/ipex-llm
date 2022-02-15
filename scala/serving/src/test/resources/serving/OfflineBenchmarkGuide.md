## Offline benchmark guide

This page contains the guide for you to run offline benchmark.

##### STEP 1:
```
git clone https://github.com/intel-analytics/bigdl.git
```

##### STEP 2:
```
cd bigdl/scala/serving/src/test/resources/serving
```

##### STEP 3 (Optional) :
```
cluster-serving-init
```
The script will generate `config.yaml` and `bigdl.jar` in the current path.

##### STEP 4:
Put the model in your local directory, and set model:/path/to/dir in `config.yaml`.

##### STEP 5:
```
offline-benchmark
```