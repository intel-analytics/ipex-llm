from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import rank, min
import random

def sql_column_api(spark):
    
    print("Start running SQL column API")
    
    # alias
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    df.select(df.age.alias("age2")).show()
    print("alias API finished")
    
    # asc
    df = spark.createDataFrame([('Tom', 80), ('Alice', None)], ["name", "height"])
    df.select(df.name).orderBy(df.name.asc()).show()
    df = spark.createDataFrame([('Tom', 80), (None, 60), ('Alice', None)], ["name", "height"])
    df.select(df.name).orderBy(df.name.asc_nulls_first()).show()
    df.select(df.name).orderBy(df.name.asc_nulls_last()).show()
    print("asc API finished")

    # between
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    df.select(df.name, df.age.between(2, 4)).show()
    print("between API finished")
    
    # bitwise
    df = spark.createDataFrame([Row(a=170, b=75)])
    df.select(df.a.bitwiseAND(df.b)).show()
    df.select(df.a.bitwiseOR(df.b)).show()
    df.select(df.a.bitwiseXOR(df.b)).show()
    print("bitwise API finished")
    
    # cast and contains
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    df.select(df.age.cast("string").alias('ages')).show()
    df.filter(df.name.contains('o')).show()
    print("cast and contains API finished")
    
    # desc
    df = spark.createDataFrame([('Tom', 80), ('Alice', None)], ["name", "height"])
    df.select(df.name).orderBy(df.name.desc()).show()
    df = spark.createDataFrame([('Tom', 80), (None, 60), ('Alice', None)], ["name", "height"])
    df.select(df.name).orderBy(df.name.desc_nulls_first()).show()
    df.select(df.name).orderBy(df.name.desc_nulls_last()).show()
    print("desc API finished")
    
    # with
    df = spark.createDataFrame([('Tom', 80), ('Alice', None)], ["name", "height"])
    df.filter(df.name.endswith('ice')).show()
    df.filter(df.name.startswith('Al')).show()
    print("with API finished")

    # eqNullSafe
    df1 = spark.createDataFrame([Row(id=1, value='foo'), Row(id=2, value=None)])
    df1.select(df1['value'] == 'foo', df1['value'].eqNullSafe('foo'), df1['value'].eqNullSafe(None)).show()
    print("eqNullSafe API finished")

    # get
    df = spark.createDataFrame([Row(r=Row(a=1, b="b"))])
    df.select(df.r.getField("b")).show()
    df = spark.createDataFrame([([1, 2], {"key": "value"})], ["l", "d"])
    df.select(df.l.getItem(0), df.d.getItem("key")).show()
    print("get API finished")

    # is
    df = spark.createDataFrame([Row(name='Tom', height=80), Row(name='Alice', height=None)])
    df.filter(df.height.isNotNull()).show()
    df.filter(df.height.isNull()).show()
    df[df.name.isin("Tom", "Mike")].show()
    print("is API finished")

    # like
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    res = df.filter(df.name.like('Al%')).collect()
    print(res)
    res = df.filter(df.name.rlike('ice$')).collect()
    print(res)
    print("like API is finished")
    
    # otherwise
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    df.select(df.name, F.when(df.age > 3, 1).otherwise(0)).show()
    print("otherwise API finished")
    
    # over
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    window = Window.partitionBy("name").orderBy("age").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df.withColumn("rank", rank().over(window)).withColumn("min", min('age').over(window)).show()
    print("over API finished")

    # substr
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    res = df.select(df.name.substr(1, 3).alias("col")).collect()
    print(res)
    print("substr API finished")
    
    # when 
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    df.select(df.name, F.when(df.age > 4, 1).when(df.age < 3, -1).otherwise(0)).show()
    print("when API finished")

    print("Finish running SQL Column API")




if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Column example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_column_api(spark)
