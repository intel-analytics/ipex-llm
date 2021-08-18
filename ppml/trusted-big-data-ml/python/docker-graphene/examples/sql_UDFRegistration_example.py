from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
import random

def sql_UDFRegistration_api(spark):
    
    print("Start running SQL UDFRegistration API")
    
    # register
    strlen = spark.udf.register("stringLengthString", lambda x: len(x))
    res = spark.sql("SELECT stringLengthString('test')").collect()
    print(res)

    _ = spark.udf.register("stringLengthInt", lambda x: len(x), IntegerType())
    res = spark.sql("SELECT stringLengthInt('test')").collect()
    print(res)

    slen = udf(lambda s: len(s), IntegerType())
    _ = spark.udf.register("slen", slen)
    res = spark.sql("SELECT slen('test')").collect()
    print(res)
     
    random_udf = udf(lambda: random.randint(0, 100), IntegerType()).asNondeterministic()
    new_random_udf = spark.udf.register("random_udf", random_udf)
    res = spark.sql("SELECT random_udf()").collect()  
    print(res)
    print("register API finished")
    
    # registerJavaFunction
    spark.udf.registerJavaFunction("javaStringLength2", "test.org.apache.spark.sql.JavaStringLength")
    res = spark.sql("SELECT javaStringLength2('test')").collect()
    print(res)
    print("registerJavaFunction API finished")

    # registerJavaUDAF
    spark.udf.registerJavaUDAF("javaUDAF", "test.org.apache.spark.sql.MyDoubleAvg")
    df = spark.createDataFrame([(1, "a"),(2, "b"), (3, "a")],["id", "name"])
    df.createOrReplaceTempView("df")
    res = spark.sql("SELECT name, javaUDAF(id) as avg from df group by name").collect()
    print(res)
    print("registerJavaUDAF API finished")

    print("Finish running SQL UDFRegistration API")





if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Dataframe example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_UDFRegistration_api(spark)
