import os
num_executors = 4  # Total number of nodes
executor_cores = 28  # Total number of vcores per node
os.environ["OMP_NUM_THREADS"] = str(executor_cores)
os.environ["KMP_BLOCKTIME"] = "200"

import time
import numpy as np
from bigdl.orca import init_orca_context, OrcaContext, stop_orca_context
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

executor_memory = "200g"
driver_cores = 4
driver_memory = "36g"
sc = init_orca_context("yarn", cores=executor_cores,
                       num_nodes=num_executors, memory=executor_memory,
                       driver_cores=driver_cores, driver_memory=driver_memory)
spark = OrcaContext.get_spark_session()

start = time.time()
df = spark.read.parquet("hdfs://172.168.0.101:8020/yahoo/id_embed.parquet")
rdd = df.rdd.repartition(num_executors)  # Each node runs one faiss task
idx_path = "/opt/flatl2.idx"


# TODO: broadcast idx?
def faiss_search(model_path, batch_size=65536, k=200): # each record: id, embedding
    def do_search(partition):
        import faiss
        faiss_idx = faiss.read_index(model_path)
        buffer = []
        for record in partition:
            if len(buffer) == batch_size:
                s1 = time.time()
                seed_ids = [row[0] for row in buffer]
                embeddings = [row[1] for row in buffer]
                buffer = [record]
                q_vec = np.stack(embeddings).astype(np.float32)
                similarity_array, idx_array = faiss_idx.search(q_vec, k=k)  # TODO: exclude itself?
                e1 = time.time()
                print("Search time: ", e1-s1)
                for i in range(batch_size):
                    for (score, rec_id) in zip(similarity_array[i], idx_array[i]):
                        yield (int(seed_ids[i]), int(rec_id), float(score))
            else:
                buffer.append(record)
        remain_size = len(buffer)
        if remain_size > 0:
            seed_ids = [row[0] for row in buffer]
            embeddings = [row[1] for row in buffer]
            q_vec = np.stack(embeddings).astype(np.float32)
            similarity_array, idx_array = faiss_idx.search(q_vec, k=k)
            for i in range(remain_size):
                for (score, rec_id) in zip(similarity_array[i], idx_array[i]):
                    yield (int(seed_ids[i]), int(rec_id), float(score))

    return do_search


res_rdd = rdd.mapPartitions(faiss_search(idx_path))
schema = StructType([
    StructField('seed_item', IntegerType(), False),
    StructField('rec_item', IntegerType(), False),
    StructField('similarity_score', FloatType(), False)
])
res_df = spark.createDataFrame(res_rdd, schema=schema)
res_df.write.mode("overwrite").parquet("hdfs://172.168.0.101:8020/yahoo/similarity_search.parquet")
end = time.time()
print("Total time used: ", end - start)
stop_orca_context()
