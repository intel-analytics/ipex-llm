from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import pandas_udf, PandasUDFType
import random

def sql_groupeddata_api(spark):
    
    print("Start running SQL GroupedData API")
    
    # agg
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    gdf = df.groupBy(df.name)
    res = sorted(gdf.agg({"*": "count"}).collect())
    print(res)
    print("agg API finished")

    # apply
    df = spark.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],("id", "v"))
    
    @pandas_udf("id long, v double", PandasUDFType.GROUPED_MAP)  
    def normalize(pdf):
        v = pdf.v
        return pdf.assign(v=(v - v.mean()) / v.std())
    
    df.groupby("id").apply(normalize).show()   
    print("apply API finished")

    # avg and count
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    res = df.groupBy().avg('age').collect()
    print(res)
    res = sorted(df.groupBy(df.age).count().collect())
    print(res)
    print("avg and count API finished")
    
    # max, mean and min
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    res = df.groupBy().max('age').collect()
    print(res)
    res = df.groupBy().mean('age').collect()
    print(res)
    res = df.groupBy().min('age').collect()
    print(res)
    print("max, mean and min API finished")

    # sum
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    res = df.groupBy().sum('age').collect()
    print(res)
    print("sum API finished")
    
    # pivot
    data = [("Banana",1000,"USA"), ("Carrots",1500,"USA"), ("Beans",1600,"USA"), \
      ("Orange",2000,"USA"),("Orange",2000,"USA"),("Banana",400,"China"), \
      ("Carrots",1200,"China"),("Beans",1500,"China"),("Orange",4000,"China"), \
      ("Banana",2000,"Canada"),("Carrots",2000,"Canada"),("Beans",2000,"Mexico")]

    df = spark.createDataFrame(data = data, schema = ["Product","Amount","Country"])
    df.printSchema()
    df.show(truncate=False) 
    res =  df.groupBy("Product").pivot("Country").sum("Amount").collect()
    print(res)
    print("pivot API finished")

    print("Finish running SQL GroupedData API")




if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL GroupedData example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_groupeddata_api(spark)
