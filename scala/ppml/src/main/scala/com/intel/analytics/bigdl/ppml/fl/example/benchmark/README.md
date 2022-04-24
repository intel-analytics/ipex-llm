# PPML Benchmark Test

## Scala API Benchmark Test

### Set up
Package PPML jar by
```bash
cd BigDL/scala/ppml &&
mvn clean package -DskipTests
```
If package fails due to some dependencies of PPML, try `cd BigDL/scala && ./make-dist.sh` to install the dependencies to local. This usually happens during the first setup and the dependencies packages are not available.

Start FLServer by
```bash
java -cp path/to/jar com.intel.analytics.bigdl.ppml.fl.FLServer
```
This starts FLServer with default config, if you need to use custom config, copy `ppml-conf.yaml` from [here]() and overwrite.

### Testing
#### FGBoost
* dataSize: size of dummy data
* dataDum: dimension of dummy data
* numRound: tree boost round
```bash
java cp path/to/jar com.intel.analytics.bigdl.ppml.fl.example.benchmark.FGBoostBenchmark --dataSize 100 --dataDim 100 --numRound 100
```

## Python API Benchmark Test
Please refer to []() to see Python API benchmark test steps.