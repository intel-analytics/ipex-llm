# Offline Recall with Faiss on Spark
This example demonstrates how to use the retrieval algorithms provided by [faiss](https://github.com/facebookresearch/faiss) 
to perform the offline recall task efficiently on Spark. 


## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster (yarn-client mode only).

Note that faiss needs to be installed via conda to get promising performance on CPU. 
Please refer to [faiss install](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for more installation guidance. 

```
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
conda install -c pytorch faiss-cpu 
pip install --pre --upgrade bigdl-friesian
```

## Generate data
You can generate some test data to run the example, which contains:
- embeddings: item embeddings of the shape [row_nums, vec_dim].
- item_dict: unique string names of items with the shape [row_nums, ].
- index data: the faiss index data generated from item embeddings.
- parquet data: items saved in parquet format, each row contains two values, the id and the embedding.

Example command:
```bash
python generate_test_data.py \
    --row_nums 200000 \
    --vec_dim 256 \
    --header_len 8 \
    --verbose \
    --index_type FlatL2 \
    --emb_path /path/to/save/vector/embeddings \
    --dict_path /path/to/save/item_dict \
    --index_save_path /path/to/save/faiss/index/data \
    --parquet_path /path/to/save/vector/embeddings/in/parquet/
```

__Options for generate_test_data:__
* `row_nums`: The number of vectors to be generated. Default to be 200000.
* `vec_dim`: The dimension of vector. Default to be 256.
* `header_len`: The header length of the unique item name. 
* `verbose`: Print more detail information. Default to be False.
* `index_type`: The faiss index_type: FlatL2 or IVFFlatL2. Default to be FlatL2.
* `emb_path`: The path to save vector embeddings. Default to be ./emb_vecs.pkl.
* `dict_path`: The path to save item_dict. Default to be ./item_dict.pkl.
* `index_save_path`: The path to save faiss index data. Default to be ./index_FlatL2.pkl.
* `parquet_path`: The path to save vector embeddings with spark, only work when use_spark is True. Default to be ./data.parquet/.

__NOTE:__ 
The file paths (like '*emb_path*') can be used directly as the corresponding input parameters for the next *search.py*.

## Search items
* Spark local, example command:
```bash
python search.py \
    --num_threads 8 \
    --cluster_mode local \
    --num_tasks 4 \
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
    --cluster_mode spark-submit \
    --num_tasks 4 \
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
    --cluster_mode yarn \
    --num_tasks 4 \
    --memory 12g \
    --top_k 100 \
    --batch_size 50000 \
    --dict_path /path/to/the/folder/of/item_dict \
    --faiss_index_path /path/to/the/folder/of/faiss/index/data \
    --parquet_path /path/to/the/folder/of/vector/embeddings/in/parquet \
    --parquet_output_path /path/to/the/folder/to/save/retrieval/results 
```

__Options for search:__
* `num_threads`: Set the environment variable OMP_NUM_THREADS for each faiss task. Default to be 8.
* `cluster_mode`: The cluster mode, one of local, spark-submit or yarn. Default to be local.
* `num_tasks`: The number of nodes to use in the cluster. Default to be 4.
* `memory`: The amount of memory to allocate on each task. Default to be 12g.
* `top_k`: The number of items to be searched for each query item. Default to be 100.
* `batch_size`: The batch size for each faiss task. Default to be 50000.
* `dict_path`: The path to item_dict.pkl. Default to be ./item_dict.pkl.
* `faiss_index_path`: The path to faiss index data. Default to be ./index_FlatL2.pkl.
* `parquet_path`: The Path to input parquet data (query items). Default to be ./data.parquet. 
* `parquet_output_path`: The path to save output parquet date (search results). Default to be ./similarity_search_L2.parquet.

__NOTE:__
When the *cluster_mode* is yarn, *dict_path*, *faiss_index_path*,
*parquet_path* and *parquet_output_path* can be HDFS paths. 
