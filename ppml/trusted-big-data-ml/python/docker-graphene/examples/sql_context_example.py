from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import SQLContext

def sql_context_api(spark):

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    print("Start running SQL context API")
    
    # createDataFrame
    
    l = [('Alice', 1)]
    sqlContext.createDataFrame(l).collect()
    res = sqlContext.createDataFrame(l, ['name', 'age']).collect()
    print(res)
    rdd = sc.parallelize(l)
    sqlContext.createDataFrame(rdd).collect()
    df = sqlContext.createDataFrame(rdd, ['name', 'age'])
    res = df.collect()
    print(res)
    print("createDataFrame API finished")

    # table and cache 
    df = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80), ('Alice', 10, 80)], ["name", "age", "height"]) 
    sqlContext.registerDataFrameAsTable(df, "table1")
    sqlContext.cacheTable("table1")
    sqlContext.uncacheTable("table1")
    sqlContext.cacheTable("table1")
    sqlContext.clearCache()
    # sqlContext.createExternalTable("table1", schema = df2)
    sqlContext.dropTempTable("table1")
    # res = df2.collect()
    # print(res)
    print("External, TempTable and cache API finished") 
    
    # getConf
    res = sqlContext.getConf("spark.sql.shuffle.partitions")
    print(res)
    res = sqlContext.getConf("spark.sql.shuffle.partitions", u"10")
    print(res)
    sqlContext.setConf("spark.sql.shuffle.partitions", u"50")
    res = sqlContext.getConf("spark.sql.shuffle.partitions", u"10")
    print(res)
    print("getConf API finished")

    # newSession
    newspark = sqlContext.newSession()
    print("newSession API finished")

    # range
    res = sqlContext.range(1, 7, 2).collect()
    print(res)
    res = sqlContext.range(3).collect()
    print(res)
    print("range API finished")
    
    # read
    res = sqlContext.read
    # text_sdf = sqlContext.readStream.text(tempfile.mkdtemp())
    # res = text_sdf.isStreaming
    # print(res)
    print("read and readStream API finished")

    # register

    
    # sql
    df = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80), ('Alice', 10, 80)], ["name", "age", "height"])
    sqlContext.registerDataFrameAsTable(df, "table1")
    df2 = sqlContext.sql("SELECT name AS f1, age as f2 from table1")
    res = df2.collect()
    print(res)
    print("sql API finished")

    # table
    df = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80), ('Alice', 10, 80)], ["name", "age", "height"])
    sqlContext.registerDataFrameAsTable(df, "table1")
    df2 = sqlContext.table("table1")
    res = (sorted(df.collect()) == sorted(df2.collect()))
    print(res)
    print("table API finished")

    # tableNames
    df = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80), ('Alice', 10, 80)], ["name", "age", "height"])
    sqlContext.registerDataFrameAsTable(df, "table1")
    res = ("table1" in sqlContext.tableNames())
    print(res)
    res = ("table1" in sqlContext.tableNames("default"))
    print(res)
    print("tableNames API finished")
    
    # tables 
    sqlContext.registerDataFrameAsTable(df, "table1")
    df2 = sqlContext.tables()
    res = df2.filter("tableName = 'table1'").first()
    print(res)
    print("tables API finished")

    print("Finish running SQL context API")






if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Context example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_context_api(spark)
