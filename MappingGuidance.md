## Mapping Guidance

This document is used to provide guidance of how BigDL and Analytics zoo code are mapping to BigDL2.0

* **BigDL**

   ***scala***

   ```spark/dl/src/main/scala/com/intel/analytics/bigdl/XYZ``` to ```scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/XYZ```

    except the following:

   ```spark/dl/src/main/scala/com/intel/analytics/bigdl/dataset``` to ```scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/feature/dataset```

   ```spark/dl/src/main/scala/com/intel/analytics/bigdl/transform``` to ```scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/feature/transform```
   
   ```spark/dl/src/main/scala/com/intel/analytics/bigdl/parameters``` to ```scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/optim/parameters```

   ```spark/dl/src/main/scala/com/intel/analytics/bigdl/python``` to ```scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/utils/python```

   ```spark/dl/src/main/scala/org/apache/spark``` to ```scala/dllib/src/main/scala/org/apache/spark```

   ```spark/spark-version``` to ```scala/common/spark-version```

   ```spark/dl/src/main/scala/com/intel/analytics/bigdl/nn/keras``` to ```scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/keras```

   ***python***

    ```pyspark/bigdl/XYZ``` to ```python/dllib/src/bigdl/dllib/XYZ```

    except the following:

   ```pyspark/bigdl/dataset``` to ```python/dllib/src/bigdl/dllib/feature/dataset```

   ```pyspark/bigdl/transform``` to ```python/dllib/src/bigdl/dllib/feature/transform```

* **Analytics Zoo**
