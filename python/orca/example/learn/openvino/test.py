from bigdl.dllib.feature.image import *
from bigdl.dllib.utils import spark
import numpy as np
from bigdl.dllib.nnframes import *
from bigdl.dllib.nn.layer import *
from pyspark.sql.functions import col, udf
from bigdl.orca.learn.openvino import Estimator
from pyspark.sql.types import ArrayType, FloatType, DoubleType, IntegerType, StringType
from bigdl.orca.data import SparkXShards
from bigdl.orca import init_orca_context, OrcaContext

# sc = init_orca_context(cluster_mode="yarn", cores=4, memory="5g",
#                        conf={"spark.driver.maxResultSize": "5g"})
sc = init_orca_context(cores=4, memory="5g", conf={"spark.driver.maxResultSize": "5g"})

spark1 = OrcaContext.get_spark_session()

rdd = sc.range(0, 16, numSlices=2)
df = rdd.map(lambda x: [x, np.random.rand(907500).tolist()]).toDF(["index", "input"])
# df.show()


def reshape(x):
	return np.array(x).reshape([3, 550, 550]).tolist()


reshape_udf = udf(reshape, ArrayType(ArrayType(ArrayType(DoubleType()))))
df = df.withColumn("input", reshape_udf(df.input))

# df.count()
# df.printSchema()
# df.show(4)
#
# # arr_images2=np.squeeze(np.array(df.select('input').collect())) #convert to numpy array and reduce the dimension
# # len(arr_images2)
# # arr_images3 = np.append(arr_images2, arr_images2,axis=0)
# # arr_images4 = np.append(arr_images3, arr_images2,axis=0)
# # len(arr_images4)
# # arr_images = np.squeeze(np.array(df.select('input').collect()))
OrcaContext._eager_mode = False
est = Estimator.from_openvino(
    model_path='/home/yina/Documents/data/myissue/openvino_model/FP32/model_float32.xml')  # load model

result_df = est.predict(df, feature_cols=["input"], batch_size=4)
# df.show()
result_df.show()
# result_df = result_df.drop("input")
#
# c = result_df.collect()
# c

# arr_images = np.squeeze(np.array(df.select('input').collect()))
# arr_images = np.append(arr_images, arr_images, axis=0)
# arr_images = np.append(arr_images, arr_images, axis=0)[:48]
# print(len(arr_images))
# result = est.predict(arr_images, batch_size=24)
# for r in result:
#         print(r.shape)

# OrcaContext._shard_size = 4
# OrcaContext._eager_mode = False
#
# from bigdl.orca.learn.utils import dataframe_to_xshards
#
# shards, _ = dataframe_to_xshards(df,
#                                  validation_data=None,
#                                  feature_cols=["input"],
#                                  label_cols=None,
#                                  mode="predict")
# # result_df = est.predict(df, feature_cols=["input"], batch_size=4)
# # s_c = shards.collect()
# result_rdd = est.predict(shards, batch_size=16)
# c = result_rdd.collect()
# c
# for a in c:
# 	for b in a:
# 		print(b.shape)

# import time
#
# time.sleep(120)
#
# arr_images2 = np.squeeze(np.array(df.select('input').collect()))
# np_r = est.predict(arr_images2)
# np_r
# result_df.printSchema()  # my model
# # res = result_df.collect()
# result_df.show(2)
# result_df2 = result_df.drop('input')  # just to decrease the size
# res2 = result_df2.collect()

# est = Estimator.from_openvino(model_path='openvino_model/FP32/model_float32.xml') #load model
# result_list2= est.predict(data=arr_images4, batch_size=4)
#
# sc = init_orca_context(cores=4, memory="5g", conf={"spark.driver.maxResultSize": "5g"})
# spark1 = OrcaContext.get_spark_session()
#
# est = Estimator.from_openvino(model_path='openvino_model/FP32/model_float32.xml') #load model
# rdd = sc.range(0, 12)
# df = rdd.map(lambda x: [x, np.random.rand(907500).tolist()]).toDF(["index", "input"]).repartition(3)
#
# # batch_size = 20
