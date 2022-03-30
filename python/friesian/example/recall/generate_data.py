import random
import numpy as np
from bigdl.orca import init_orca_context, OrcaContext, stop_orca_context
from pyspark.sql.types import Row
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType

sc = init_orca_context(cores="*", memory="125g", conf={"spark.driver.maxResultSize": "10g"})
spark = OrcaContext.get_spark_session()
rdd = sc.parallelize(range(6000000))

def gen_vector(x):
    return (x, np.random.rand(128).astype(np.float32) + random.randint(0, 1000)*random.random())


rdd = rdd.map(gen_vector)
rdd.cache()  # If not cache, will run random generation again when saving to parquet.
data = rdd.collect()
vectors = np.array([row[1] for row in data], dtype=np.float32)

np.save("data.npy", vectors)

schema = StructType([
    StructField('id', IntegerType(), False),
    StructField('embedding', ArrayType(FloatType()), False)
])

df = spark.createDataFrame(rdd.map(lambda x: Row(x[0], x[1].tolist())), schema=schema)
df.write.parquet("data.parquet")
print("Finished")
stop_orca_context()