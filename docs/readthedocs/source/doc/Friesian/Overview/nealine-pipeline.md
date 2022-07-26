# Friesian Nearline Pipelines

## Overview

The Friesian Nearline Pipeline encapsulates kv store initialization, faiss index generation and near real-time updates.

1. KV store initialization

   Load user/item parquet files and keep them in the Key-Value store. Load user embedding parquet file and keep it in the Key-Value store.

2. Faiss index generation

   Retrieve item embedding vectors and build the faiss index.

3. Near real-time updates (This feature will be released in a future version)

   Make updates to the features, user embeddings and faiss index from time to time.

## Key Concepts

### Faiss

[Faiss](https://github.com/facebookresearch/faiss) is a library for efficient similarity search. The Recall Service uses intel optimized faiss to search similar candidates (100 ~ 1000 candidates) from millions of items. 

HNSWlibInt16_32 is used in the Recall Service.

### Redis

[Redis](https://redis.io/) stands for Remote Dictionary Server, is an open source, in-memory data store. It provides kv store and high performance data access in the Feature Service.

#### The schema of redis tables
The user features, item features and user embedding vectors are saved in Redis.
The data saved in Redis is a key-value set.

##### Key in Redis
The key in Redis consists of 3 parts: key prefix, data type, and data id. 
- Key prefix is `redisKeyPrefix` specified in the feature service config file. 
- Data type is one of `user` or `item`. 
- Data id is the value of `userIDColumn` or `itemIDColumn`.
Here is an example of key: `2tower_user:29`

##### Value in Redis
A row in the input parquet file will be converted to java array of object, then serialized into byte array, and encoded into Base64 string.

##### Data schema entry
Every key prefix and data type combination has its data schema entry to save the corresponding column names. The key of the schema entry is `keyPrefix + dataType`, such as `2tower_user`. The value of the schema entry is a string of column names separated by `,`, such as `enaging_user_follower_count,enaging_user_following_count,enaging_user_is_verified`.

### Config Files

Each initializer has its config template for providing important information, and users can fill it in and pass it to the initializers using `-c config.yaml`.

### Recall Initializer

The Recall Initializer loads item embeddings and generate the faiss index which will be loaded in the online serving Recall Service.

#### Recall Initializer Config Template

The config template for the Recall Initializer is [config_recall.yaml](https://github.com/intel-analytics/BigDL/blob/main/scala/friesian/src/main/resources/nearlineConfig/config_recall.yaml).

You can uncomment the parameters you need and modify the values.

```yaml
##### RecallInitializer Config

# default: 128, the dimensionality of the embedding vectors
# indexDim: 

# default: null, path to saved index path, must be provided
indexPath: ./item_50.idx

# default: null, must be provided, item id column name
itemIDColumn: tweet_id

# default: null, must be provided, item embedding column name
itemEmbeddingColumn: prediction

# default: null, must be provided, the path to the item embedding file, only support parquet file
initialDataPath: /path/to/the/item/embedding/file.parquet
```

- indexDim: the dimensionality of the embedding vectors. Default: 128
- indexPath: path to saved index path, must be provided. Default: null.
- itemIDColumn: item id column name, must be provided. Default: null.
- itemEmbeddingColumn: item embedding column name, must be provided. Default: null.
- initialDataPath: the path to the item embedding file, only support parquet file, must be provided. Default: null.

#### The Schema of the Input Parquet Files

The embedding parquet files should contain at least 2 columns, id column and prediction column. The id column should be IntegerType and the column name should be specified in the config files. The prediction column should be DenseVector type, and you can transfer your existing embedding vectors using pyspark:
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.ml.linalg import VectorUDT, DenseVector

spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

df = spark.read.parquet("data_path")

def trans_densevector(data):
   return DenseVector(data)

vector_udf = udf(lambda x: trans_densevector(x), VectorUDT())
# suppose the embedding column (ArrayType(FloatType,true)) is the existing user/item embedding.
df = df.withColumn("prediction", vector_udf(col("embedding")))
df.write.parquet("output_file_path", mode="overwrite")
```


### Feature Initializer

The Feature Initializer loads user & item feature and user embeddings into the redis kv store. Then the online Feature Service connects to the redis and searches features from it.

#### Feature Initializer Config Template

The config template for the Feature Initializer is [config_feature.yaml](https://github.com/intel-analytics/BigDL/blob/main/scala/friesian/src/main/resources/nearlineConfig/config_feature.yaml).

You can uncomment the parameters you need and modify the values.

```yaml
##### FeatureInitializer Config

# default: "", prefix for redis key
redisKeyPrefix:

# default: 0, item slot type on redis cluster. 0 means slot number use the default value 16384, 1 means all keys save to same slot, 2 means use the last character of id as hash tag.
# redisClusterItemSlotType: 

# default: null, one of initialUserDataPath or initialItemDataPath must be
# provided. Only support parquet file
initialUserDataPath: /path/to/the/user/feature/file.parquet
initialItemDataPath: /path/to/the/item/feature/file.parquet

# default: null, if initialUserDataPath != null, userIDColumn and userFeatureColumns must be provided.
userIDColumn: 
userFeatureColumns: 

# default: null, if loadInitialData=true and initialItemDataPath != null, userIDColumn and
# userFeatureColumns must be provided
itemIDColumn: 
itemFeatureColumns: 

### Redis Configuration
# default: localhost:6379
# redisUrl: 

# default: standalone, redis server type, can be "standalone", "sentinel", or "cluster"
# redisType:

# default: null, must be provided if redisType=sentinel
# redisSentinelMasterName:

# default: localhost:26379
# redisSentinelMasterURL:
```

- redisKeyPrefix: prefix for redis key. Default: "".
- redisClusterItemSlotType: item slot type on redis cluster. 0 means slot number use the default value 16384, 1 means all keys save to same slot, 2 means use the last character of id as hash tag. Only used when `redisType=cluster`. Default: 0.
- initialUserDataPath: the path to the user feature parquet file. Note that one of initialUserDataPath or initialItemDataPath must be provided. Default: null.
- initialItemDataPath: the path to the item feature parquet file. Note that one of initialUserDataPath or initialItemDataPath must be provided. Default: null.
- userIDColumn: user id column name. If initialUserDataPath != null, userIDColumnmust be provided. Default: null.
- userFeatureColumns: user feature column names. If initialUserDataPath != null, userFeatureColumns must be provided. Default: null.
- itemIDColumn: item id column name. If initialItemDataPath != null, itemIDColumnmust be provided. Default: null.
- itemFeatureColumns: item feature column names. If initialItemDataPath != null, itemFeatureColumns must be provided. Default: null.
- redisUrl: redisUrl for redis standalone and redis cluster. If redisType=sentinel, redisUrl will be ignored. Default: localhost:6379.
- redisType: redis server type, can be "standalone", "sentinel", or "cluster".
- redisSentinelMasterName: redis sential master name, it must be provided if redisType=sentinel. Default: null.
- redisSentinelMasterURL: redis sential master url, it must be provided if redisType=sentinel. Default: localhost:26379.


#### The Schema of the Input Parquet Files

The feature parquet files should contain at least 2 columns, the id column and other feature columns. The feature columns can be int, float, double, long and array of int, float, double and long. Here is an example of the WideAndDeep model feature, tweet_id is the id column.

```bash
+-------------+--------+--------+----------+--------------------------------+---------------------------------+------------+-----------+---------+----------------------+-----------------------------+
|present_media|language|tweet_id|tweet_type|engaged_with_user_follower_count|engaged_with_user_following_count|len_hashtags|len_domains|len_links|present_media_language|engaged_with_user_is_verified|
+-------------+--------+--------+----------+--------------------------------+---------------------------------+------------+-----------+---------+----------------------+-----------------------------+
|            9|      43|     924|         2|                               6|                                3|         0.0|        0.1|      0.1|                    45|                            1|
|            0|       6| 4741724|         2|                               3|                                3|         0.0|        0.0|      0.0|                   527|                            0|
+-------------+--------+--------+----------+--------------------------------+---------------------------------+------------+-----------+---------+----------------------+-----------------------------+
```


## Pull Friesian Serving Docker Image

Users can pull the Friesian Serving docker image using `docker pull intelanalytics/friesian-serving`

## Start Recall Initializer


1. Prepare item embedding parquet file

You should use the trained 2-tower model to predict the item embeddings, and save the results as parquet file.

2. Prepare config file [config_recall.yaml](#recall-initializer-config-template): modify `itemIDColumn`, `itemEmbeddingColumn` and `initialDataPath` according to your embedding file location and column names. **Note** that we use the bind mount to mount the current directory into /opt/work/mnt in the container, so paths in the config file should start with `mnt/`.

3. Your file structure will like:
   ```
   └── $(pwd)
    ├── item_emb.parquet
    └── nearline
        └── config_recall.yaml
   ```
4. Build item embedding faiss index using `docker run -it --net host --rm -v $(pwd):/opt/work/mnt intelanalytics/friesian-serving recall-init -c mnt/nearline/config_recall.yaml` and you will see the file `item.idx` under your working directory.

## Start Feature Initializer

1. Prepare redis 
   1. Download redis and start the redis server using `redis-server`
   2. Disable protected mode using `redis-cli set protected-mode no`
2. Prepare data files & config files
   1. Prepare feature data parquet files
      1. You should preprocess the user and item datasets and save the preprocessed features as parquet files. For example, `wnd_user.parquet` and `wnd_item.parquet`.
   2. Prepare user embedding parquet files
      1. You should use the trained 2-tower model to predict the user embeddings, and save the results as parquet file. For example, `user_emb.parquet`.
   3. Prepare config file [config_feature.yaml](#feature-initializer-config-template): modify `initialUserDataPath`, `initialItemDataPath`, `userIDColumn`, `userFeatureColumns`, `itemIDColumn` and `itemFeatureColumns` according to your feature file location and feature names. **Note** that we use the bind mount to mount the current directory into /opt/work/mnt in the container, so paths in the config file should start with `mnt/`.
   4. Prepare config file [config_feature_vec.yaml](#feature-initializer-config-template): modify `initialUserDataPath`, `userIDColumn` and `userFeatureColumns` according to your embedding file location and column names. **Note** that we use the bind mount to mount the current directory into /opt/work/mnt in the container, so paths in the config file should start with `mnt/`.
   5. Your file structure will like:
   ```
   └── $(pwd)
    ├── wnd_user.parquet
    ├── wnd_item.parquet
    ├── user_emb.parquet
    └── nearline
        ├── config_feature.yaml
        └── config_feature_vec.yaml
   ```
3. Load user & item features into redis using `docker run -it --net host --rm -v $(pwd):/opt/work/mnt intelanalytics/friesian-serving feature-init -c mnt/nearline/config_feature.yaml`
4. Load user embeddings into redis using `docker run -it --net host --rm -v $(pwd):/opt/work/mnt intelanalytics/friesian-serving feature-init -c mnt/nearline/config_feature_vec.yaml`
