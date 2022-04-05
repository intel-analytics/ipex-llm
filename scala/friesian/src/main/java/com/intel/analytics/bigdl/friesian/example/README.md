# Recommend items using Friesian-Serving Framewrok

This example shows how to recommend items for each item based on item to item similarity.

## steps to run this example

1. Prepare features and embeddings
   Follow example of Friesian to train a two tower model, generate item embeddings.
2. Follow instructions[https://github.com/intel-analytics/BigDL/tree/main/scala/friesian#quick-start
   ] to Pull docker image and start container
3. Load item embeddings into redis
```
export SERVING_JAR_PATH=bigdl-friesian-spark_2.4.6-2.1.0-SNAPSHOT-serving.jar
export OMP_NUM_THREADS=1
echo "Starting loading initial features......"
java -Dspark.master=local[*] -cp $SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.nearline.feature.FeatureInitializer -c /opt/work/similarity/config_feature_simi.yaml

```
4. Build and load faiss index
```
echo "Starting initializing recall index......"
java -Dspark.master=local[*] -cp $SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.nearline.recall.RecallInitializer  -c /opt/work/similarity/config_recall_simi.yaml ```
```
5. Start feature service
```
echo "Starting feature service......"
java -cp $SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.serving.feature.FeatureServer -c /opt/work/similarity/config_feature_simi.yaml 
```
6. Start recall service
```
echo "Starting recall service......"
java -cp $SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.serving.recall.RecallServer -c /opt/work/similarity/config_recall_simi.yaml 
```
7. Start SimilarityClient
```
echo "Starting similarity client......"
java -Dspark.master=local[*] -cp $SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.example.SimilarityClient -target localhost:8084 -dataDir /opt/work/similarity/item_ebd.parquet
```




