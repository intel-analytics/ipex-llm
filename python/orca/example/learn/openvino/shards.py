import numpy as np
from bigdl.orca.learn.openvino import Estimator
from bigdl.orca.data import SparkXShards
from bigdl.orca import init_orca_context, OrcaContext

sc = init_orca_context(cores=4, memory="5g", conf={"spark.driver.maxResultSize": "5g"})

rdd = sc.range(0, 5, numSlices=4)
bs = 4


def gen_data(d):
	return {"x": np.random.random([bs, 3, 550, 550])}


s1 = SparkXShards(rdd)
s1 = s1.transform_shard(gen_data)
est = Estimator.from_openvino(
	model_path='/home/yina/Documents/data/myissue/openvino_model/FP32/model_float32.xml')

result_rdd = est.predict(s1)
a = result_rdd.collect()
print(a)
