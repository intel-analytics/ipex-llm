# Contribute to Cluster Serving

This is the guide to contribute your code to Cluster Serving.

Cluster Serving takes advantage of Analytics Zoo core with integration of Deep Learning Frameworks, e.g. Tensorflow, OpenVINO, PyTorch, and implements the inference logic on top of it, and parallelize the computation with Flink and Redis by default. To contribute more features to Cluster Serving, you could refer to following sections accordingly.
## Dev Environment

### Get Code and Prepare Branch
Go to Analytics Zoo main repo https://github.com/intel-analytics/analytics-zoo, press Fork to your github repo, and git clone the forked repo to local. Use `git checkout -b your_branch_name` to create a new branch, and you could start to write code and pull request to Analytics Zoo from this branch.
### Environment Set up
You could refer to [Analytics Zoo Scala Developer Guide](https://analytics-zoo.readthedocs.io/en/latest/doc/UserGuide/develop.html#scala) to set up develop environment. Cluster Serving is an Analytics Zoo Scala module.
### Debug in IDE
Once set up, you could copy the `/path/to/analytics-zoo/scripts/cluster-serving/config.yaml` to `/path/to/analytics-zoo/config.yaml`, and run `zoo/src/main/scala/com/intel/analytics/zoo/serving/ClusterServing.scala` in IDE. Since IDE consider `/path/to/analytics-zoo/` as the current directory, it would read the config file in it.

Running `ClusterServing.scala` in IDE would behave the same as you run Cluster Serving locally, so once started, you could run python client code to finish an end-to-end test just as you run Cluster Serving in [Programming Guide](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ClusterServingGuide/ProgrammingGuide.md#4-model-inference).


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
