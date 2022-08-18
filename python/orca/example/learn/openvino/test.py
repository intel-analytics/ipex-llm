from bigdl.dllib.nn.layer import *
from pyspark.sql.functions import col, udf
from bigdl.orca.learn.openvino import Estimator
from pyspark.sql.types import ArrayType, FloatType, DoubleType
from bigdl.orca import init_orca_context, OrcaContext

sc = init_orca_context(cores=4, memory="5g", conf={"spark.driver.maxResultSize": "5g"})

spark1 = OrcaContext.get_spark_session()

rdd = sc.range(0, 32, numSlices=2)
df = rdd.map(lambda x: [x, np.random.rand(907500).tolist()]).toDF(["index", "input"])
# df.show()


def reshape(x):
	return np.array(x).reshape([3, 550, 550]).tolist()


reshape_udf = udf(reshape, ArrayType(ArrayType(ArrayType(DoubleType()))))
df = df.withColumn("input", reshape_udf(df.input))

# df.count()
# df.printSchema()
# df.show(4)

OrcaContext._eager_mode = False
est = Estimator.from_openvino(
	model_path='/home/yina/Documents/data/myissue/openvino_model/FP32/model_float32.xml')  # load model

# --------------- dataframe
# result_df = est.predict(df, feature_cols=["input"], batch_size=4)
# # df.show()
# # result_df.show()
# result_df = result_df.drop("input")
# c = result_df.collect()
# c

# --------------- xshards
OrcaContext._shard_size = 4
OrcaContext._eager_mode = False

from bigdl.orca.learn.utils import dataframe_to_xshards

shards, _ = dataframe_to_xshards(df,
                                 validation_data=None,
                                 feature_cols=["input"],
                                 label_cols=None,
                                 mode="predict")
result_rdd = est.predict(shards, batch_size=4)
c = result_rdd.collect()
c
for a in c:
	for b in a:
		print(b.shape)
	print("--------------")

# ------------------- ndarray
# arr_images = np.squeeze(np.array(df.select('input').collect()))
# arr_images = np.append(arr_images, arr_images, axis=0)
# arr_images = np.append(arr_images, arr_images, axis=0)[:50]
# result_np = est.predict(arr_images)
# for nd in result_np:
# 	print(nd.shape)

import time

time.sleep(120)
