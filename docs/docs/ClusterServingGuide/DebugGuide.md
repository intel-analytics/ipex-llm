# Cluster Serving Debug Guide

This guide provides step-by-step debug to help you if you are stuck when using Cluster Serving
### Check if Cluster Serving environment is ready
Run following commands in terminal
```
echo $FLINK_HOME
echo $REDIS_HOME
```
the output directory
```
/path/to/flink-version
/path/to/redis-version
``` 
 
should be displayed, otherwise, go to [Programming Guide](ProgrammingGuide.md) **Installation** section.

### Check if Flink Cluster is working
Run following commands in terminal
```
netstat -tnlp
```
output like following should be displayed, `6123,8081` is Flink default port usage.
```
tcp6       0      0 :::6123                 :::*                    LISTEN      xxxxx/java
tcp6       0      0 :::8081                 :::*                    LISTEN      xxxxx/java
```
if not, run `$FLINK_HOME/bin/start-cluster.sh` to start Flink cluster.

After that, check Flink log in `$FLINK_HOME/log/`, check the log file of `flink-xxx-standalone-xxx.log` and `flink-xxx-taskexecutor-xxx.log` to make sure there is no error.

If the port could not bind in this step, kill the program which use the port, and `$FLINK_HOME/bin/stop-cluster.sh && $FLINK_HOME/bin/start-cluster.sh` to restart Flink cluster.
### Check if Cluster Serving is running
```
$FLINK_HOME/bin/flink list
```
output of Cluster Serving job information should be displayed, if not, go to [Programming Guide](ProgrammingGuide.md) **Launching Service** section to make sure you call `cluster-serving-start` correctly.

### Still, I get no result
Getting no result means after you call 1.`InputQueue.enqueue` and use `OutputQueue.query`, or 2. just use `InputQueue.predict`  to get result, you get `[]` or `NaN`

If you get empty result, aka `[]`, means your input get no return, you could go to `$FLINK_HOME/log/flink-taskexecutor-...log` to check your output. If you are sure you have done above check and still get empty result, raise issue [here](https://github.com/intel-analytics/analytics-zoo/issues) and post this log.

If you get invalid result, aka `NaN`, means your input does not match your model, please check your data shape. e.g. if you use Tensorflow Keras Model, you can use `model.predict(data)` locally to check if it works. This test also applies to other deep learning frameworks.


### Troubleshooting

1. `Duplicate registration of device factory for type XLA_CPU with the same priority 50`

This error is caused by Flink ClassLoader. Please put cluster serving related jars into `${FLINK_HOME}/lib`.

2. `servable Manager config dir not exist`

Check if `servables.yaml` exists in current directory. If not, download from [github](https://github.com/intel-analytics/analytics-zoo/blob/master/ppml/trusted-realtime-ml/scala/docker-graphene/servables.yaml).
