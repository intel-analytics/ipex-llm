BigDL need some environment variables be set correctly to get a good performance. Engine.init method
can help you set and verify them.

## How to do it in the code?
Before any BigDL related code piece, please invoke `Engine.init` like this

```scala
val conf = Engine.createSparkConf()
val sc = new SparkContext(conf)
Engine.init
```

```python
conf=create_spark_conf()
sc = SparkContext(conf)
init_engine()
```

Please note that there's an old Engine.init(executorNum, coreNumber) API, which need you pass in the
executor number and core number. As user may input an incorrect value, we have changed to auto
detect these two values. So the old API is deprecated.

## What if the spark context has been created before my code get executed?
In some platform or application(e.g. spark-shell, pyspark or jupyter notebook), the spark context
is created before your code execution. In such case, you cannot use the 'createSparkConf' API to
initialize your own spark context. Such platform or application should always allow you to modify
the spark configuration in some way, so you can pass in the required spark configurations by youself.

### What's the required configurations?
You can find them in the `conf/spark-bigdl.conf` file.

### How to do it?
If you use spark shell or pyspark notebook

```shell
# Spark shell
spark-shell --properties-file dist/conf/spark-bigdl.conf ...

# Pyspark
pyspark --properties-file dist/conf/spark-bigdl.conf ... 
```

In your code
```scala
Engine.init 
```
```python
init_engine()
```

## Run BigDL without Spark
If you run BigDL models without Spark, you should set the JVM property `bigdl.localMode` to true.
So the Engine.init won't check spark related environments.
