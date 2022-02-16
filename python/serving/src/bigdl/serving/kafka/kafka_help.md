# Kafka Guide
This is a guide to use python to run kafka.

You can use `kafka_example.py` to test the kafka environment and get basic usages of kafka by python.

## Install Kafka
To setup the kafka environment, install kafka from [Kafka Download](https://kafka.apache.org/downloads).

## Install Python Package
To use kafka in python, run `pip install kafka-python`.
    
## Test With Example
### Initial Setup (Start Kafka)
First go to your kafka folder `cd <pathToKafka>`.

With default settings, you can start the zookeeper by
```bash
bin/zookeeper-server-start.sh config/zookeeper.properties
```

Then start the kafka by   
```bash
bin/kafka-server-start.sh config/server.properties
```

### Run python test
Now use `kafka_example.py` to run a simple test. `kafka_example.py` includes constructors of both a **Producer** and a **Consumer**. You can first build a consumer with
```python
python kafka_examples.py consumer_demo
```
this will open up a consumer listening for messages from a kafka topic (initially no output).

Then go to a new terminal and run
```python
python kafka_examples.py producer_demo
```
to build a producer and send some sample messages to the kafka topic. You should see the output
```bash
send 0
send 1
send 2
```
Now switch back to the consumer terminal, you should see the messages sent by the producers have already been received by the consumers
```bash
receive, key: test, value: 0
receive, key: test, value: 1
receive, key: test, value: 2
```
**Make sure to keep the topic set in `producer_demo` and `consumer_demo` the same so that the consumer side can receive the messages sent by producer side.**

### Run KafkaServing
#### Start CluserServing with Kafka
If you want to test ClusterServing with Kafka, change [ClusterServing.scala](https://github.com/intel-analytics/bigdl/blob/master/bigdl/src/main/scala/com/intel/analytics/bigdl/serving/ClusterServing.scala) in `bigdl/blob/master/bigdl/src/main/scala/com/intel/analytics/bigdl/serving/ClusterServing.scala` as follows:

```scala
streamingEnv.addSource(new FlinkRedisSource(helper)) -> streamingEnv.addSource(new FlinkKafkaSource(helper))
```
```scala
  .addSink(new FlinkRedisSink(helper)) ->   .addSink(new FlinkKafkaSink(helper))
```
then you can start ClusterServing as usual.
#### Test with python
You can use python to test KafkaServing, similar to [python test](#run-python-test). Change the topic (producer) in `kafka_example.py` to `serving_stream` and the topic (consumer) in `kafka_example.py` to `cluster-serving_serving_stream`. Modify the message sent by the producer to match the input format for serving and run the producer&consumer. You should be seeing predict results in consumer terminal.

**Make sure to serialize and deserialize the data on Kafka using the same serializer/deserializer** --- For example, on python producer side, the serializer should match the deserializer in `bigdl/blob/master/bigdl/src/main/scala/com/intel/analytics/bigdl/serving/engine/FlinkKafkaSource.scala`.
