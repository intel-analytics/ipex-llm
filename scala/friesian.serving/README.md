## Friesian gRPC Recommendation Framework


### Quick Start
You can run Friesion gRPC Recommendation Framework using the official Docker images.

First you can follow the following steps to run the WnD demo.

1. Pull docker image from dockerhub
```bash
The first version will be uploaded soon
```

2. Run & enter docker container
```bash
docker run -itd --name grpcwnd1 --net=host grpcwnd
docker exec -it grpcwnd1 bash
```

3. Add vec_feature_user_prediction.parquet, vec_feature_item_prediction.parquet, wnd model,
   wnd_item.parquet and wnd_user.parquet

4. Start ranking service
```bash
export OMP_NUM_THREADS=1
export TF_DISABLE_MKL=1
java -cp bigdl-friesian-serving-spark_2.4.6-0.14.0-SNAPSHOT.jar com.intel.analytics.bigdl.friesian.serving.ranking.RankingServer -c config_ranking.yaml > logs/inf.log 2>&1 &
```

5. Start feature service for recommender service
```bash
./redis-5.0.5/src/redis-server &
java -Dspark.master=local[*] -cp bigdl-friesian-serving-spark_2.4.6-0.14.0-SNAPSHOT.jar com.intel.analytics.bigdl.friesian.serving.feature.FeatureServer -c config_feature.yaml > logs/feature.log 2>&1 &
```

6. Start feature service for recall service
```bash
java -Dspark.master=local[*] -cp bigdl-friesian-serving-spark_2.4.6-0.14.0-SNAPSHOT.jar com.intel.analytics.bigdl.friesian.serving.feature.FeatureServer -c config_feature_vec.yaml > logs/fea_recall.log 2>&1 &
```

7. Start recall service
```bash
java -Dspark.master=local[*] -Dspark.driver.maxResultSize=2G -cp bigdl-friesian-serving-spark_2.4.6-0.14.0-SNAPSHOT.jar com.intel.analytics.bigdl.friesian.serving.recall.RecallServer -c config_recall.yaml > logs/vec.log 2>&1 &
```

8. Start recommender service
```bash
java -cp bigdl-friesian-serving-spark_2.4.6-0.14.0-SNAPSHOT.jar com.intel.analytics.bigdl.friesian.serving.recommender.RecommenderServer -c config_recommender.yaml > logs/rec.log 2>&1 &
```

9. Check if the services are running
```bash
ps aux|grep friesian
```
You will see 5 processes start with 'java'

10. Run client to test
```bash
java -Dspark.master=local[*] -cp bigdl-friesian-serving-spark_2.4.6-0.14.0-SNAPSHOT.jar com.intel.analytics.bigdl.friesian.serving.recommender.RecommenderMultiThreadClient -target localhost:8980 -dataDir wnd_user.parquet -k 50 -clientNum 4 -testNum 2
```
11. Close services
```bash
ps aux|grep friesian (find the service pid)
kill xxx (pid of the service which should be closed)
```

### Config for different service
You can pass some important information to services using `-c config.yaml`
```bash
java -Dspark.master=local[*] -Dspark.driver.maxResultSize=2G -cp bigdl-friesian-serving-spark_2.4.6-0.14.0-SNAPSHOT.jar com.intel.analytics.bigdl.friesian.serving.recall.RecallServer -c config_recall.yaml
```

#### Ranking Service Config
Config with example:
```yaml
# Default: 8980, which port to create the server
servicePort: 8083

# Default: 0, open a port for prometheus monitoring tool, if set, user can check the
# performance using prometheus
monitorPort: 1234

# model path must be provided
modelPath: /home/yina/Documents/model/recys2021/wnd_813/recsys_wnd

# default: null, savedmodel input list if the model is tf savedmodel. If not provided, the inputs
# of the savedmodel will be arranged in alphabetical order
savedModelInputs: serving_default_input_1:0, serving_default_input_2:0, serving_default_input_3:0, serving_default_input_4:0, serving_default_input_5:0, serving_default_input_6:0, serving_default_input_7:0, serving_default_input_8:0, serving_default_input_9:0, serving_default_input_10:0, serving_default_input_11:0, serving_default_input_12:0, serving_default_input_13:0

# default: 1, number of models used in inference service
modelParallelism: 4
```

##### Feature Service Config
Config with example:
1. load data into redis. Search data from redis
```yaml
### Basic setting
# Default: 8980, which port to create the server
servicePort: 8082

# Default: null, open a port for prometheus monitoring tool, if set, user can check the
# performance using prometheus
monitorPort: 1235

# 'kv' or 'inference' default: kv
serviceType: kv

# default: false, if need to load initial data to redis, set true
loadInitialData: true

# default: "", prefix for redis key
redisKeyPrefix:

# default: null, if loadInitialData=true, initialUserDataPath or initialItemDataPath must be
# provided. Only support parquet file
initialUserDataPath: /home/yina/Documents/data/recsys/preprocess_output/wnd_user.parquet
initialItemDataPath: /home/yina/Documents/data/recsys/preprocess_output/wnd_exp1/wnd_item.parquet

# default: null, if loadInitialData=true and initialUserDataPath != null, userIDColumn and
# userFeatureColumns must be provided
userIDColumn: enaging_user_id
userFeatureColumns: enaging_user_follower_count,enaging_user_following_count

# default: null, if loadInitialData=true and initialItemDataPath != null, userIDColumn and
# userFeatureColumns must be provided
itemIDColumn: tweet_id
itemFeatureColumns: present_media, language, tweet_id, hashtags, present_links, present_domains, tweet_type, engaged_with_user_follower_count,engaged_with_user_following_count, len_hashtags, len_domains, len_links, present_media_language, tweet_id_engaged_with_user_id

# default: null, user model path or item model path must be provided if serviceType
# contains 'inference'. If serviceType=kv, usermodelPath, itemModelPath and modelParallelism will
# be ignored
# userModelPath: 

# default: null, user model path or item model path must be provided if serviceType
# contains 'inference'. If serviceType=kv, usermodelPath, itemModelPath and modelParallelism will
# be ignored
# itemModelPath: 

# default: 1, number of models used for inference
# modelParallelism: 

### Redis Configuration
# default: localhost:6379
# redisUrl:

# default: 256, JedisPoolMaxTotal
# redisPoolMaxTotal:
```

2. load user features into redis. Get features from redis, use model at 'userModelPath' to do
   inference and get the user embedding
```yaml
### Basic setting
# Default: 8980, which port to create the server
servicePort: 8085

# Default: null, open a port for prometheus monitoring tool, if set, user can check the
# performance using prometheus
monitorPort: 1236

# 'kv' or 'inference' default: kv
serviceType: kv, inference

# default: false, if need to load initial data to redis, set true
loadInitialData: true

# default: ""
redisKeyPrefix: 2tower_

# default: null, if loadInitialData=true, initialDataPath must be provided. Only support parquet
# file
initialUserDataPath: /home/yina/Documents/data/recsys/preprocess_output/guoqiong/vec_feature_user.parquet
# initialItemDataPath: 

# default: null, if loadInitialData=true and initialUserDataPath != null, userIDColumn and
# userFeatureColumns must be provided
#userIDColumn: user
userIDColumn: enaging_user_id
userFeatureColumns: user

# default: null, if loadInitialData=true and initialItemDataPath != null, userIDColumn and
# userFeatureColumns must be provided
# itemIDColumn: 
# itemFeatureColumns: 

# default: null, user model path or item model path must be provided if serviceType
# includes 'inference'. If serviceType=kv, usermodelPath, itemModelPath and modelParallelism will
# be ignored
userModelPath: /home/yina/Documents/model/recys2021/2tower/guoqiong/user-model

# default: null, user model path or item model path must be provided if serviceType
# contains 'inference'. If serviceType=kv, usermodelPath, itemModelPath and modelParallelism will
# be ignored
# itemModelPath: 

# default: 1, number of models used for inference
# modelParallelism: 

### Redis Configuration
# default: localhost:6379
# redisUrl:

# default: 256, JedisPoolMaxTotal
# redisPoolMaxTotal:
```

#### Recall Service Config
Config with example:

1. load initial item vector from vec_feature_item.parquet and item-model to build faiss index.
```yaml
# Default: 8980, which port to create the server
servicePort: 8084

# Default: null, open a port for prometheus monitoring tool, if set, user can check the
# performance using prometheus
monitorPort: 1238

# default: false, if load saved index, set true
# loadSavedIndex: true

# default: false, if true, the built index will be saved to indexPath. Ignored when
# loadSavedIndex=true
saveBuiltIndex: true

# default: null, path to saved index path, must be provided if loadSavedIndex=true
indexPath: ./2tower_item_full.idx

# default: false
getFeatureFromFeatureService: true

# default: localhost:8980, feature service target
featureServiceURL: localhost:8085

itemIDColumn: tweet_id
itemFeatureColumns: item

# default: null, user model path must be provided if getFeatureFromFeatureService=false
# userModelPath: 

# default: null, item model path must be provided if loadSavedIndex=false and initialDataPath is
# not orca predict result
itemModelPath: /home/yina/Documents/model/recys2021/2tower/guoqiong/item-model

# default: null,  Only support parquet file
initialDataPath: /home/yina/Documents/data/recsys/preprocess_output/guoqiong/vec_feature_item.parquet

# default: 1, number of models used in inference service
modelParallelism: 1
```

2. load existing faiss index
```yaml
# Default: 8980, which port to create the server
servicePort: 8084

# Default: null, open a port for prometheus monitoring tool, if set, user can check the
# performance using prometheus
monitorPort: 1238

# default: false, if load saved index, set true
loadSavedIndex: true

# default: null, path to saved index path, must be provided if loadSavedIndex=true
indexPath: ./2tower_item_full.idx

# default: false
getFeatureFromFeatureService: true

# default: localhost:8980, feature service target
featureServiceURL: localhost:8085

# itemIDColumn: 
# itemFeatureColumns: 

# default: null, user model path must be provided if getFeatureFromFeatureService=false
# userModelPath: 

# default: null, item model path must be provided if loadSavedIndex=false and initialDataPath is
# not orca predict result
# itemModelPath: 

# default: null,  Only support parquet file
# initialDataPath: 

# default: 1, number of models used in inference service
# modelParallelism: 
```
#### Recommender Service Config
Config with example:

```yaml
 Default: 8980, which port to create the server
 servicePort: 8980

 # Default: null, open a port for prometheus monitoring tool, if set, user can check the
 # performance using prometheus
 monitorPort: 1237

 # default: null, must be provided, item column name
 itemIDColumn: tweet_id
 
# default: null, must be provided, column names for inference, order related.
inferenceColumns: present_media_language, present_media, tweet_type, language, hashtags, present_links, present_domains, tweet_id_engaged_with_user_id, engaged_with_user_follower_count, engaged_with_user_following_count, enaging_user_follower_count, enaging_user_following_count, len_hashtags, len_domains, len_links

# default: localhost:8980, recall service target
recallServiceURL: localhost:8084

# default: localhost:8980, feature service target
featureServiceURL: localhost:8082

# default: localhost:8980, inference service target
rankingServiceURL: localhost:8083
```

### Run Java Client

#### Generate proto java files
You should init a maven project and use proto files in [friesian gRPC project](https://github.com/analytics-zoo/friesian/tree/recsys-grpc/src/main/proto)
Make sure to add the following extensions and plugins in your pom.xml, and replace
*protocExecutable* with your own protoc executable.
```xml
    <build>
        <extensions>
            <extension>
                <groupId>kr.motd.maven</groupId>
                <artifactId>os-maven-plugin</artifactId>
                <version>1.6.2</version>
            </extension>
        </extensions>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.8.0</version>
                <configuration>
                    <source>8</source>
                    <target>8</target>
                </configuration>
            </plugin>
            <plugin>
                <groupId>org.xolstice.maven.plugins</groupId>
                <artifactId>protobuf-maven-plugin</artifactId>
                <version>0.6.1</version>
                <configuration>
                    <protocArtifact>com.google.protobuf:protoc:3.12.0:exe:${os.detected.classifier}</protocArtifact>
                    <pluginId>grpc-java</pluginId>
                    <pluginArtifact>io.grpc:protoc-gen-grpc-java:1.37.0:exe:${os.detected.classifier}</pluginArtifact>
                    <protocExecutable>/home/yina/Documents/protoc/bin/protoc</protocExecutable>
                </configuration>
                <executions>
                    <execution>
                        <goals>
                            <goal>compile</goal>
                            <goal>compile-custom</goal>
                        </goals>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
```
Then you can generate the gRPC files with
```bash
mvn clean install
```
#### Call recommend service function using blocking stub
You can check the [Recommend service client example](https://github.com/analytics-zoo/friesian/blob/recsys-grpc/src/main/java/grpc/recommend/RecommendClient.java) on Github

```java
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.*;

public class RecommendClient {
    public static void main(String[] args) {
        // Create a channel
        ManagedChannel channel = ManagedChannelBuilder.forTarget(targetURL).usePlaintext().build();
        // Init a recommend service blocking stub
        RecommenderGrpc.RecommenderBlockingStub blockingStub = RecommenderGrpc.newBlockingStub(channel);
        // Construct a request
        int[] userIds = new int[]{1};
        int candidateNum = 50;
        int recommendNum = 10;
        RecommendRequest.Builder request = RecommendRequest.newBuilder();
        for (int id : userIds) {
            request.addID(id);
        }
        request.setCandidateNum(candidateNum);
        request.setRecommendNum(recommendNum);
        RecommendIDProbs recommendIDProbs = null;
        try {
            recommendIDProbs = blockingStub.getRecommendIDs(request.build());
            logger.info(recommendIDProbs.getIDProbListList());
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
        }
    }
}
```

### Run Python Client
Install the python packages listed below (you may encounter [pyspark error](https://stackoverflow.com/questions/58700384/how-to-fix-typeerror-an-integer-is-required-got-type-bytes-error-when-tryin) if you have python>=3.8 installed, try to downgrade to python<=3.7 and try again).
```bash
pip install jupyter notebook==6.1.4 grpcio grpcio-tools pandas fastparquet pyarrow
```
After you activate your server successfully, you can

#### Generate proto python files
Generate the files with
```bash
python -m grpc_tools.protoc -I../../protos --python_out=<path_to_output_folder> --grpc_python_out=<path_to_output_folder> <path_to_friesian>/src/main/proto/*.proto
```

#### Call recommend service function using blocking stub
You can check the [Recommend service client example](https://github.com/analytics-zoo/friesian/blob/recsys-grpc/Serving/WideDeep/recommend_client.ipynb) on Github
```python
# create a channel
channel = grpc.insecure_channel('localhost:8980')
# create a recommend service stub
stub = recommender_pb2_grpc.RecommenderStub(channel)
request = recommender_pb2.RecommendRequest(recommendNum=10, candidateNum=50, ID=[36407])
results = stub.getRecommendIDs(request)
print(results.IDProbList)

```
### Scale-out for Big Data
#### Redis Cluster
For large data set, Redis standalone has no enough memory to store whole data set, data sharding and Redis cluster are supported to handle it. You only need to set up a Redis Cluster to get it work.

First, start N Redis instance on N machines.
```
redis-server --cluster-enabled yes --cluster-config-file nodes-0.conf --cluster-node-timeout 50000 --appendonly no --save "" --logfile 0.log --daemonize yes --protected-mode no --port 6379
```
on each machine, choose a different port and start another M instances(M>=1), as the slave nodes of above N instances.

Then, call initialization command on one machine, if you choose M=1 above, use `--cluster-replicas 1`
```
redis-cli --cluster create 172.168.3.115:6379 172.168.3.115:6380 172.168.3.116:6379 172.168.3.116:6380 172.168.3.117:6379 172.168.3.117:6380 --cluster-replicas 1
```
and the Redis cluster would be ready.

#### Scale Service with Envoy
Each of the services could be scaled out. It is recommended to use the same resource, e.g. single machine with same CPU and memory, to test which service is bottleneck. From empirical observations, vector search and inference usually be.

##### How to run envoy:
1. [download](https://www.envoyproxy.io/docs/envoy/latest/start/install) and deploy envoy(below use docker as example):
 * download: `docker pull envoyproxy/envoy-dev:21df5e8676a0f705709f0b3ed90fc2dbbd63cfc5`
2. run command: `docker run --rm -it  -p 9082:9082 -p 9090:9090 envoyproxy/envoy-dev:79ade4aebd02cf15bd934d6d58e90aa03ef6909e --config-yaml "$(cat path/to/service-specific-envoy.yaml)" --parent-shutdown-time-s 1000000`
3. validate: run `netstat -tnlp` to see if the envoy process is listening to the corresponding port in the envoy config file.
4. For details on envoy and sample procedure, read [envoy](envoy.md).
