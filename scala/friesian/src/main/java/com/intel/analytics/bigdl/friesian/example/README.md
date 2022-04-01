# Recommend items using Friesian-Serving API

This example shows how to recommend items for each item in a list based on item to item similarity.

## steps to run this exmaple

1. Prepare Data and model
    Follow example of Friesian to train a two tower model, generate item embeddings.
2. Load item embeddings into redis and build faiss index
    Load item embeddings into redis 
```
bash start_service.sh feature-init -c nfs/guoqiong/similarity/config_simi_feature_init.yaml
```
3. build item embedding into faiss index
 ```
 bash start_service.sh recall-init -c nfs/guoqiong/similarity/config_simi_recall_init.yaml
 ```
4. Start Feature Service
```
 bash start_service.sh feature -c nfs/guoqiong/config_simi_feature.yaml > feature.log 2>&1 &
```
5. Start recall service
```
 bash start_service.sh feature -c nfs/guoqiong/config_simi_recall.yaml > recall.log 2>&1 &
```
7. Start SimilarityClient
```
start_service.sh client -target localhost:8980 -dataDir nfs/guoqiong/wnd_item.parquet -k 50
```




