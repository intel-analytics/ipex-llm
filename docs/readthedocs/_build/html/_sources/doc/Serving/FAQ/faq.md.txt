# Cluster Serving FAQ

## General Debug Guide
You could use following guide to debug if serving is not working properly.

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



### Troubleshooting

1. `Duplicate registration of device factory for type XLA_CPU with the same priority 50`

This error is caused by Flink ClassLoader. Please put cluster serving related jars into `${FLINK_HOME}/lib`.

2. `servable Manager config dir not exist`

Check if `servables.yaml` exists in current directory. If not, download from [github](https://github.com/intel-analytics/bigdl/blob/master/ppml/trusted-realtime-ml/scala/docker-graphene/servables.yaml).
### Still, I get no result
If you still get empty result, raise issue [here](https://github.com/intel-analytics/bigdl/issues) and post the output/log of your serving job.
