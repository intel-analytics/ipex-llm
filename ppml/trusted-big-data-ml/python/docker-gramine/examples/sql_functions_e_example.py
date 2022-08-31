from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def sql_functions_e_example(spark):

    # element_at
    df = spark.createDataFrame([(["a", "b", "c"],), ([],)], ['data'])
    df.select(element_at(df.data, 1)).show()
    print("element_at API finished")

    # encode
    df = spark.createDataFrame([(1, 'a'), (2, 'b'), (3, 'c')], ["n1", "s1"])
    df.select(encode(df.s1, "utf-8")).show()
    print("encode API finished")

    # exp
    df = spark.createDataFrame([(1, 'a'), (2, 'b'), (3, 'c')], ["n1", "s1"])
    df.select(exp(df.n1)).show()
    print("exp API finished")

    # explode
    from pyspark.sql import Row
    eDF = spark.createDataFrame([Row(a=1, intlist=[1,2,3], mapfield={"a": "b"})])
    eDF.select(explode(eDF.intlist).alias("anInt")).show()
    eDF.select(explode(eDF.mapfield).alias("key", "value")).show()
    print("explode API finished")

    # explode_outer
    df = spark.createDataFrame([(1, ["foo", "bar"], {"x": 1.0}), (2, [], {}), (3, None, None)], ("id", "an_array", "a_map"))
    df.select("id", "an_array", explode_outer("a_map")).show()
    print("explode_outer API finished")

    # expm1
    df = spark.createDataFrame([(1, 'a'), (2, 'b'), (3, 'c')], ["n1", "s1"])
    df.select(expm1(df.n1)).show()
    print("expm1 API finished")

    # expr
    df = spark.createDataFrame([('Alice', 21), ('Bob', 23)], ["name", "age"])
    df.select(expr("length(name)")).show()
    df.select(length("name")).show()
    print("expr API finished")
    
    print("Finish running function_e API")

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL functions_e API example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    sql_functions_e_example(spark)
    spark.stop()
