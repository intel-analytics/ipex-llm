# Recall items by using faiss
This example demonstrates how to use BigDL Friesian 
to get the recalled items by using the retrieval algorithms provided by faiss.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster (yarn-client mode only).
Also Need to install faiss via conda to get reasonable performance on CPU, referring [faiss install](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

```
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
conda install -c pytorch faiss-cpu 
pip install --pre --upgrade bigdl-friesian
```

## Generate data
You can generate some test data for testing, example command:
```bash
python generate_test_data.py \
    --row_nums 200000 \
    --vec_dim 256 \
    --verbose \
    --index_type FlatL2 \
    --emb_path /path/to/save/vector/embeddings \
    --dict_path /path/to/save/item_dict \
    --index_save_path /path/to/save/faiss/index/data \
    --parquet_path /path/to/save/vector/embeddings/in/parquet/
```

## Retrieving items
* Spark local, example command:
```bash
python search.py \
    --num_threads 8 \
    --num_repartition 4 \
    --cluster_mode local \
    --top_k 100 \
    --batch_size 50000 \
    --dict_path /path/to/the/folder/of/item_dict \
    --faiss_index_path /path/to/the/folder/of/faiss/index/data \
    --parquet_path /path/to/the/folder/of/vector/embeddings/in/parquet \
    --parquet_output_path /path/to/the/folder/to/save/retrieval/results 
```

* Spark spark-submit, example command:
```bash
python search.py \
    --num_threads 8 \
    --num_repartition 4 \
    --cluster_mode spark-submit \
    --top_k 100 \
    --batch_size 50000 \
    --dict_path /path/to/the/folder/of/item_dict \
    --faiss_index_path /path/to/the/folder/of/faiss/index/data \
    --parquet_path /path/to/the/folder/of/vector/embeddings/in/parquet \
    --parquet_output_path /path/to/the/folder/to/save/retrieval/results 
```

* Spark yarn client mode, example command:
```bash
python search.py \
    --num_threads 8 \
    --num_repartition 12 \
    --cluster_mode yarn \
    --num_nodes 4 \
    --executor_cores 8 \
    --executor_memory 50g \
    --top_k 100 \
    --batch_size 50000 \
    --dict_path /path/to/the/folder/of/item_dict \
    --faiss_index_path /path/to/the/folder/of/faiss/index/data \
    --parquet_path /path/to/the/folder/of/vector/embeddings/in/parquet \
    --parquet_output_path /path/to/the/folder/to/save/retrieval/results 
```

__Options for generate_test_data:__
* `row_nums`: The number of vectors to be generated. Default to be 200000.
* `vec_dim`: The dimension of vector. Default to be 256.
* `verbose`: Print more detail information. Default to be False.
* `index_type`: The faiss index_type: FlatL2 or IVFFlatL2. Default to be FlatL2.
* `emb_path`: The path to save vector embeddings. Default to be ./emb_vecs.pkl.
* `dict_path`: The path to save item_dict. Default to be ./item_dict.pkl.
* `index_save_path`: The path to save faiss index data. Default to be ./index_FlatL2.pkl.
* `parquet_path`: The path to save vector embeddings with spark, only work when use_spark is True. Default to be ./data.parquet/.

__Options for search:__
* `num_threads`: Set the environment variable OMP_NUM_THREADS for each faiss task. Default to be 8.
* `num_repartition`: The number of repartition. Default to be 12.
* `cluster_mode`: The cluster mode, one of local, spark-submit or yarn. Default to be local.
* `num_nodes`: The number of nodes to use in the cluster. Default to be 4.
* `executor_cores`: The number of cores to use on each node. Default to be 8.
* `executor_memory`: The amount of memory to allocate on each node. Default to be 50g.
* `top_k`: The number of items to be searched for each query item. Default to be 100.
* `batch_size`: The batch size for each faiss task. Default to be 50000.
* `dict_path`: The path to item_dict.pkl. Default to be ./item_dict.pkl.
* `faiss_index_path`: The path to faiss index data. Default to be ./index_FlatL2.pkl.
* `parquet_path`: The Path to input parquet data (query items). Default to be ./data.parquet. 
* `parquet_output_path`: The path to save output parquet date (search results). Default to be ./similarity_search_L2.parquet.
