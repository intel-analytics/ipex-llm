from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.util.common import *
from bigdl.ml_pipeline.dl_classifier import *
from pyspark.sql.types import *
from pyspark.context import SparkContext

if __name__ == "__main__":
    sc = SparkContext(appName="DLEstimatorMultiLabelLR", conf=create_spark_conf().setMaster("local[1]"))
    sqlContext = SQLContext(sc)
    init_engine()
    model = Sequential().add(Linear(2, 2))
    criterion = MSECriterion()
    estimator = DLEstimator(model, criterion, [2], [2]).setBatchSize(4).setMaxEpoch(10)
    data = sc.parallelize([
        ((2.0, 1.0), (1.0, 2.0)),
        ((1.0, 2.0), (2.0, 1.0)),
        ((2.0, 1.0), (1.0, 2.0)),
        ((1.0, 2.0), (2.0, 1.0))])

    schema = StructType([
        StructField("features", ArrayType(DoubleType(), False), False),
        StructField("label", ArrayType(DoubleType(), False), False)])
    df = sqlContext.createDataFrame(data, schema)
    dlModel = estimator.fit(df)
    dlModel.transform(df).show(False)
    sc.stop()