import time
import numpy as np
from bigdl.orca import init_orca_context, OrcaContext, stop_orca_context
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

sc = init_orca_context(cores="*", memory="125g", conf={"spark.driver.maxResultSize": "10g"})
spark = OrcaContext.get_spark_session()

start = time.time()
df = spark.read.parquet("data.parquet").repartition(8) # coalesce
rdd = df.rdd.filter(lambda x: x[0] < 10000)
idx_path = "flatl2.idx"

# faiss index is ~3G and if there are too many partitions, may easily run OOM.
# TODO: can broadcast index?
def faiss_search(model_path, batch_size=65536, k=200): # each record: id, embedding
    def do_search(partition):
        import faiss
        faiss_idx = faiss.read_index(model_path)
        buffer = []
        for record in partition:
            if len(buffer) == batch_size:
                seed_ids = [row[0] for row in buffer]
                embeddings = [row[1] for row in buffer]
                buffer = [record]
                q_vec = np.stack(embeddings).astype(np.float32)
                similarity_array, idx_array = faiss_idx.search(q_vec, k=k)  # TODO: exclude itself
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
            similarity_array, idx_array = faiss_idx.search(q_vec, k=k)  # TODO: exclude itself
            for i in range(remain_size):
                for (score, rec_id) in zip(similarity_array[i], idx_array[i]):
                    yield (int(seed_ids[i]), int(rec_id), float(score))

    return do_search

res_rdd = rdd.mapPartitions(faiss_search(idx_path))
# print(res_rdd.count())
# print(res_rdd.take(5))
# rows = rdd.take(10)
# res = list(faiss_search(idx_path)(rows))

schema = StructType([
    StructField('seed_item', IntegerType(), False),
    StructField('rec_item', IntegerType(), False),
    StructField('similarity_score', FloatType(), False)
])
res_df = spark.createDataFrame(res_rdd, schema=schema)
res_df.write.mode("overwrite").parquet("similarity_search.parquet")
# print(res_df.count())

end = time.time()
print("Total time used: ", end - start)
stop_orca_context()
