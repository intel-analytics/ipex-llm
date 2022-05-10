# PPML FGBoost Benchmark test

## Start FLServer
To run benchmark test, first start FLServer. Copy the PPML jar to current working directory and run
```
bash ppml/scripts/start-fl-server.sh
```
Alternatively, starting from java or python is also supported
```
# java cmd, PPML jar need to be provided
java -cp $PPML_JAR com.intel.analytics.bigdl.ppml.fl.FLServer

# python cmd, python source code and environment variable PYTHONPATH/BIGDL_CLASSPATH need to be set
python python/ppml/src/bigdl/ppml/fl/fl_server.py
```
## Prepare the client configuration file
After FLServer starts, a config file should be provided if default config needs to be overwritten. The default config would connect FLServer at `localhost:8980`.

A config file is provided in `ppml/ppml-conf.yaml`, copy it to current working directory and client would use it when client starts.
## Start benchmark test

### Dummy data benchmark
In dummy data benchmark test, the random data would be generated and size, dimension of data could be set

Start the FGBoost Regression with dummy data with arguments
* data size: the size of dummy data, default 100
* data dim: the dimension of dummy data, default 10
* num round: the number of boosting round, default 10

e.g. start the benchmark test with data size 10000, data dimension 100, boosting round 5
```
# java cmd
java -cp $PPML_JAR com.intel.analytics.bigdl.ppml.fl.example.benchmark.FGBoostDummyDataBenchmark --dataSize 10000 --dataDim 100 --numRound 5

# python cmd
python python/ppml/example/benchmark/fgboost_benchmark/fgboost_dummy_data_benchmark.py --data_size 10000 --data_dim 100 --num_round 5
```
### Real data benchmark
In real data benchmark test, the train and test data path should be provided, and number of copies could be set, e.g. if set 10, the test would copy the data to 10 times of origin.

Start the Start the FGBoost Regression with read data with arguments
* train path: the path of train data
* test path: the path of test data
* data size: the number of copies of data
* num round: the number of boosting round, default 10

e.g. a preprocessed House Prices data is stored in `scala/ppml/demo/data`, to start the benchmark test with it, with 10 times data copies, boosting round 5
```
# java cmd
java -cp $PPML_JAR com.intel.analytics.bigdl.ppml.fl.example.benchmark.FGBoostRealDataBenchmark 
    --trainPath scala/ppml/demo/data/house-prices-train-preprocessed.csv
    --testPath scala/ppml/demo/data/house-prices-test-preprocessed.csv
    --dataSize 10 --numRound 5
    
# python cmd
python python/ppml/example/benchmark/fgboost_benchmark/fgboost_dummy_data_benchmark.py 
    --train_path scala/ppml/demo/data/house-prices-train-preprocessed.csv
    --test_path scala/ppml/demo/data/house-prices-test-preprocessed.csv
    --data_size 10 --num_round 5
```