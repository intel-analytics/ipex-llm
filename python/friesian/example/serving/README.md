# Recommend items using Friesian-Serving Framework

This example demonstrates item similarity recommendation using Friesian.

## steps to run this example
1. Prepare environment
```
pip install grpcio
pip install protobuf
```
2. Prepare features and embeddings
   Follow the example [here](../two_tower) to train a two tower model and generate item embeddings.
3. Follow [instructions](https://github.com/intel-analytics/BigDL/tree/main/scala/friesian#quick-start) to Pull docker image and start container
4. Load item embeddings into redis
```
export SERVING_JAR_PATH=bigdl-friesian-spark_2.4.6-2.1.0-SNAPSHOT-serving.jar
export OMP_NUM_THREADS=1
echo "Starting loading initial features......"
java -Dspark.master=local[*] -cp $SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.nearline.feature.FeatureInitializer -c /opt/work/similarity/config_feature_simi.yaml

```
4. Build and load faiss index
```
echo "Starting initializing recall index......"
java -Dspark.master=local[*] -cp $SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.nearline.recall.RecallInitializer  -c /opt/work/similarity/config_recall_simi.yaml
```
5. Start feature service
```
echo "Starting feature service......"
java -cp $SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.serving.feature.FeatureServer -c /opt/work/similarity/config_feature_simi.yaml > feature.log
```
6. Start recall service
```
echo "Starting recall service......"
java -cp $SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.serving.recall.RecallServer -c /opt/work/similarity/config_recall_simi.yaml > recall.log
```
7. Run Similarity_client
```
echo "Starting similarity client......"
python  similarity_client.py  --target yourhost:8084 --data_dir ../item_ebd.parquet
```




