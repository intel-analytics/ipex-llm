# Start Cluster Serving
  
## Launching Service of Serving

Before do inference (predict), you have to start serving service. This section shows how to start/stop the service. 

### Start
You can use following command to start Cluster Serving.
```
cluster-serving-start
```

Normally, when calling `cluster-serving-start`, your `config.yaml` should be in current directory. You can also use `cluster-serving-start -c config_path` to pass config path `config_path` to Cluster Serving manually.

### Stop
You can use Flink UI in `localhost:8081` by default, to cancel your Cluster Serving job.

Or you can use `${FLINK_HOME}/bin/flink list` to get serving job ID and call `${FLINK_HOME|/bin/flink cancel $ID`.

### Shut Down
You can use following command to shutdown Cluster Serving. This operation will stop all Cluster Serving jobs and Redis server. Note that your data in Redis will be removed when you shutdown. 
```
cluster-serving-shutdown
```
If you are using Docker, you could also run `docker rm` to shutdown Cluster Serving.
### Start Multiple Serving
To run multiple Cluster Serving job, e.g. the second job name is `serving2`, then use following configuration
```
# model path must be provided
# modelPath: /path/to/model

# name, default is serving_stream, you need to specify if running multiple servings
# jobName: serving2
```
then call `cluster-serving-start` in this directory would start another Cluster Serving job with this new configuration.

Then, in Python API, pass `name=serving2` argument during creating object, e.g.
```
input_queue=InputQueue(name=serving2)
output_queue=OutputQueue(name=serving2)
```
Then the Python API would interact with job `serving2`.

### HTTP Server
If you want to use sync API for inference, you should start a provided HTTP server first. User can submit HTTP requests to the HTTP server through RESTful APIs. The HTTP server will parse the input requests and pub them to Redis input queues, then retrieve the output results and render them as json results in HTTP responses.

#### Prepare
User can download a bigdl-${VERSION}-http.jar from the Nexus Repository with GAVP: 
```
<groupId>com.intel.analytics.bigdl</groupId>
<artifactId>bigdl-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}</artifactId>
<version>${ZOO_VERSION}</version>
```
User can also build from the source code:
```
mvn clean package -P spark_2.4+ -Dmaven.test.skip=true
```
#### Start the HTTP Server
User can start the HTTP server with following command.
```
java -jar bigdl-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ZOO_VERSION}-http.jar
```
And check the status of the HTTP server with:
```
curl  http://${BINDED_HOST_IP}:${BINDED_HOST_PORT}/
```
If you get a response like "welcome to BigDL web serving frontend", that means the HTTP server is started successfully.
#### Start options
User can pass options to the HTTP server when start it:
```
java -jar bigdl-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ZOO_VERSION}-http.jar --redisHost="172.16.0.109"
```
All the supported parameter are listed here:
* **interface**: the binded server interface, default is "0.0.0.0"
* **port**: the binded server port, default is 10020
* **redisHost**: the host IP of redis server, default is "localhost"
* **redisPort**: the host port of redis server, default is 6379
* **redisInputQueue**: the input queue of redis server, default is "serving_stream"
* **redisOutputQueue**: the output queue of redis server, default is "result:" 
* **parallelism**: the parallelism of requests processing, default is 1000
* **timeWindow**: the timeWindow wait to pub inputs to redis, default is 0
* **countWindow**: the timeWindow wait to ub inputs to redis, default is 56
* **tokenBucketEnabled**: the switch to enable/disable RateLimiter, default is false
* **tokensPerSecond**: the rate of permits per second, default is 100
* **tokenAcquireTimeout**: acquires a permit from this RateLimiter if it can be obtained without exceeding the specified timeout(ms), default is 100

**User can adjust these options to tune the performance of the HTTP server.**
