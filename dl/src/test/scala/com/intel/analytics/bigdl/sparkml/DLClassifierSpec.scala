package com.intel.analytics.bigdl.sparkml

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{GreyImgNormalizer, GreyImgToBatch, SampleToGreyImg}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.lenet.Utils._
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{AnyType, DLClassifier}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

class DLClassifierSpec extends FlatSpec with Matchers{

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "DLClassifier for MNIST" should "get good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = new SparkContext("local[1]", "DLClassifier")
    val sqlContext = new SQLContext(sc)

    // read test data
    val resource = getClass().getClassLoader().getResource("mnist")
    val validationData = Paths.get(processPath(resource.getPath()) + File.separator, "t10k-images.idx3-ubyte")
    val validationLabel = Paths.get(processPath(resource.getPath()) + File.separator, "t10k-labels.idx1-ubyte")
    val batchSize = 10
    val classNum = 10
    val model = LeNet5(classNum)

    val validationSet = DataSet.array(load(validationData, validationLabel))
      .transform(SampleToGreyImg(28, 28))
    val normalizerVal = GreyImgNormalizer(validationSet)
    val valSet = validationSet.transform(normalizerVal)
      .transform(GreyImgToBatch(batchSize))
    val valData = valSet.data(looped = false)

    // init
    val valTrans = new DLClassifier()
      .setInputCol("features")
      .setOutputCol("predict")

    // set paramMap
    val schema = StructType(StructField(valTrans.getInputCol, AnyType, true) :: Nil)
    val params = ParamMap(valTrans.modelTrain -> model)
    val tensorBuffer = new ArrayBuffer[Tensor[Float]]()
    var res: DataFrame = null

    while (valData.hasNext) {
      val batch = valData.next()
      val input = batch.data

      var i = 1
      while (i <= input.size(1)) {
        tensorBuffer.append(input.select(1, i))
        i += 1
      }

      val rowRDD = sc.parallelize((tensorBuffer)).map(p => Row(p))
      val testData = sqlContext.createDataFrame(rowRDD, schema)
      res = valTrans.transform(testData, params)
      res.show()

      tensorBuffer.clear()
    }
    sc.stop()
  }
}

