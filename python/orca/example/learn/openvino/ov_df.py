from bigdl.dllib.nn.layer import *
from pyspark.sql.functions import col, udf
from bigdl.orca.learn.openvino import Estimator
from pyspark.sql.types import ArrayType, FloatType, DoubleType
from bigdl.orca import init_orca_context, OrcaContext

sc = init_orca_context(cores=4, memory="5g", conf={"spark.driver.maxResultSize": "5g"})

spark1 = OrcaContext.get_spark_session()

rdd = sc.range(0, 48, numSlices=2)
df = rdd.map(lambda x: [x, np.random.rand(907500).tolist()]).toDF(["index", "input"])


def reshape(x):
	return np.array(x).reshape([3, 550, 550]).tolist()


reshape_udf = udf(reshape, ArrayType(ArrayType(ArrayType(DoubleType()))))
df = df.withColumn("input", reshape_udf(df.input))
# df = rdd.map(lambda x: [x]).toDF(["index"])
df.cache()
df.count()

# df.printSchema()
# df.show(4)

OrcaContext._eager_mode = False
est = Estimator.from_openvino(
	model_path='/home/yina/Documents/data/myissue/openvino_model/FP32/model_float32.xml')  # load model

# rdf = est.predict_without_ov(df, batch_size=8)
# cols = rdf.columns
# c = rdf.collect()
# rdf.write.parquet("/home/yina/Documents/data/myissue/reshape.parquet")
# rdf.show()

result_df = est.predict(df, feature_cols=["input"], batch_size=4)
# # result_df = result_df.drop("input")
c = result_df.collect()
# # df.show()
# result_df.show()
# c
print("end")

import time

time.sleep(120)