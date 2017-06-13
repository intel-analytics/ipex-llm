
## **Overview**

BigDL provides `DLClassifier` for users with Apache Spark MLlib experience, which
provides high level API for training a BigDL Model with the Apache Spark `Transfomer`
pattern, thus users can conveniently fit BigDL into a ML pipeline and do prediction.

Currently only scala interface are implemented for `DLClassifier`.

---
## **DLClassifier**

`DLClassifier` extends `org.apache.spark.ml.Transformer` and supports model prediction from
Apache Spark DataFrame/Dataset. 

To use `DLClassifier` for prediction, user should specify

* the model structure constructed from BigDL layers. You can also use some predefined model
like LetNet or ResNet.
* batch shape, defined as `Array(batchsize, featuresize)`. Internally the feature data are converted to BigDL tensors, to predict more efficiently.

**Scala example:**
```scala
package com.intel.analytics.bigdl.example.imageclassification

import java.nio.file.Paths
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.example.imageclassification.MlUtils._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.DLClassifier
import org.apache.spark.sql.SQLContext

/**
 * An example to show how to use DLClassifier Transform
 */
object ImagePredictor {
  def main(args: Array[String]): Unit = {
    predictParser.parse(args, new PredictParams()).map(param => {
      val conf = Engine.createSparkConf()
      conf.setAppName("Predict with trained model")
      val sc = new SparkContext(conf)
      Engine.init
      val sqlContext = new SQLContext(sc)

      val partitionNum = Engine.nodeNumber() * Engine.coreNumber()
      val model = loadModel(param)
      val valTrans = new DLClassifier().setInputCol("features").setOutputCol("predict")

      val paramsTrans = ParamMap(
        valTrans.modelTrain -> model,
        valTrans.batchShape ->
        Array(param.batchSize, 3, imageSize, imageSize))

      // load image set from local
      val paths = LocalImageFiles.readPaths(Paths.get(param.folder), hasLabel = false)
      val valRDD = sc.parallelize(imagesLoad(paths, 256), partitionNum)

      val transf = RowToByteRecords() ->
          BytesToBGRImg() ->
          BGRImgCropper(imageSize, imageSize) ->
          BGRImgNormalizer(testMean, testStd) ->
          BGRImgToImageVector()

      val valDF = transformDF(sqlContext.createDataFrame(valRDD), transf)

      valTrans.transform(valDF, paramsTrans)
      sc.stop()
    })
  }
}
```