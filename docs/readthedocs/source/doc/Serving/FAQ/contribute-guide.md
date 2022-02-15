# Contribute to Cluster Serving

This is the guide to contribute your code to Cluster Serving.

Cluster Serving takes advantage of BigDL core with integration of Deep Learning Frameworks, e.g. Tensorflow, OpenVINO, PyTorch, and implements the inference logic on top of it, and parallelize the computation with Flink and Redis by default. To contribute more features to Cluster Serving, you could refer to following sections accordingly.
## Dev Environment

### Get Code and Prepare Branch
Go to BigDL main repo https://github.com/intel-analytics/bigdl, press Fork to your github repo, and git clone the forked repo to local. Use `git checkout -b your_branch_name` to create a new branch, and you could start to write code and pull request to BigDL from this branch.
### Environment Set up
You could refer to [BigDL Scala Developer Guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/develop.html#scala) to set up develop environment. Cluster Serving is an BigDL Scala module.

### Debug in IDE
Cluster Serving depends on Flink and Redis. To install Redis and start Redis server,
```
$ export REDIS_VERSION=5.0.5
$ wget http://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz && \
    tar xzf redis-${REDIS_VERSION}.tar.gz && \
    rm redis-${REDIS_VERSION}.tar.gz && \
    cd redis-${REDIS_VERSION} && \
    make
$ ./src/redis-server
```
in IDE, embedded Flink would be used so that no dependency is needed.

Once set up, you could copy the `/path/to/bigdl/scripts/cluster-serving/config.yaml` to `/path/to/bigdl/config.yaml`, and run `zoo/src/main/scala/com/intel/analytics/zoo/serving/ClusterServing.scala` in IDE. Since IDE consider `/path/to/bigdl/` as the current directory, it would read the config file in it.

Run `zoo/src/main/scala/com/intel/analytics/zoo/serving/http/Frontend2.scala` if you use HTTP frontend.
 
Once started, you could run python client code to finish an end-to-end test just as you run Cluster Serving in [Programming Guide](https://github.com/intel-analytics/bigdl/blob/master/docs/docs/ClusterServingGuide/ProgrammingGuide.md#4-model-inference).
### Test Package
Once you write the code and complete the test in IDE, you can package the jar and test.

To package,
```
cd /path/to/bigdl/zoo
./make-dist.sh
```
Then, in `target` folder, copy `bigdl-xxx-flink-udf.jar` to your test directory, and rename it as `zoo.jar`, and also copy the `config.yaml` to your test directory.

You could copy `/path/to/bigdl/scripts/cluster-serving/cluster-serving-start` to start Cluster Serving, this scripts will start Redis server for you and submit Flink job. If you prefer not to control Redis, you could use the command in it `${FLINK_HOME}/bin/flink run -c com.intel.analytics.zoo.serving.ClusterServing zoo.jar` to start Cluster Serving.

To run frontend, call `java -cp zoo.jar com.intel.analytics.zoo.serving.http.Frontend2`.

The rest are the same with test in IDE.

## Add Features
### Data Connector
Data connector is the producer of Cluster Serving. The remote clients put data into data pipeline
#### Scala code (The Server)

To define a new data connector to, e.g. Kafka, Redis, or other database, you have to define a Flink Source first.

You could refer to `com/intel/analytics/zoo/serving/engine/FlinkRedisSource.scala` as an example.

```
class FlinkRedisSource(params: ClusterServingHelper)
  extends RichParallelSourceFunction[List[(String, String)]] {
  @volatile var isRunning = true

  override def open(parameters: Configuration): Unit = {
    // initlalize the connector
  }

  override def run(sourceContext: SourceFunction
    .SourceContext[List[(String, String)]]): Unit = while (isRunning) {
    // get data from data pipeline
  }

  override def cancel(): Unit = {
    // close the connector
  }
}
```
Then you could refer to `com/intel/analytics/zoo/serving/engine/FlinkInference.scala` as the inference method to your new connector. Usually it could be directly used without new implementation. However, you could still define your new method if you need.

Finally, you have to define a Flink Sink, to write data back to data pipeline.

You could refer to `com/intel/analytics/zoo/serving/engine/FlinkRedisSink.scala` as an example.

```
class FlinkRedisSink(params: ClusterServingHelper)
  extends RichSinkFunction[List[(String, String)]] {
  
  override def open(parameters: Configuration): Unit = {
    // initialize the connector
  }

  override def close(): Unit = {
    // close the connector
  }

  override def invoke(value: List[(String, String)], context: SinkFunction.Context[_]): Unit = {
    // write data to data pipeline
  }
}
```
Please note that normally you should do the space (memory or disk) control of your data pipeline in your code.


Please locate Flink Source and Flink Sink code to `com/intel/analytics/zoo/serving/engine/`

If you have some method which need to be wrapped as a class, you could locate them in `com/intel/analytics/zoo/serving/pipeline/`
#### Python Code (The Client)
You could refer to `pyzoo/zoo/serving/client.py` to define your client code according to your data connector.

Please locate this part of code in `pyzoo/zoo/serving/data_pipeline_name/`, e.g. `pyzoo/zoo/serving/kafka/` if you create a Kafka connector.
##### put to data pipeline
It is recommended to refer to `InputQueue.enqueue()` and `InputQueue.predict()` method. This method calls `self.data_to_b64` method first and add data to data pipeline. You could define a similar enqueue method to work with your data connector.
##### get from data pipeline
It is recommended to refer to `OutputQueue.query()` and `OutputQueue.dequeue()` method. This method gets result from data pipeline and calls `self.get_ndarray_from_b64` method to decode. You could define a similar dequeue method to work with your data connector.

## Benchmark Test
You could use `zoo/src/main/scala/com/intel/analytics/zoo/serving/engine/Operations.scala` to test the inference time of your model. 

The script takes two arguments, run it with `-m modelPath` and `-j jsonPath` to indicate the path to the model and the path to the prepared json format operation template of the model.

The model will output the inference time stats of preprocessing, prediction and postprocessing processes, which varies with the different preprocessing/postprocessing time and thread numbers.
