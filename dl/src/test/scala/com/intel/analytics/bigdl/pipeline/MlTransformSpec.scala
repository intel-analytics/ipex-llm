package com.intel.analytics.bigdl.pipeline

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{GreyImgNormalizer, GreyImgToBatch, SampleToGreyImg}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.lenet.Utils._
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{AnyType, MlTransform}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class MlTransformSpec extends FlatSpec with Matchers{

  "MlTransform for MNIST" should "get good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = new SparkContext("local[1]", "MlTransform")
    val sqlContext = new SQLContext(sc)

    // Generate test data
    val input1: Array[Double] = Array(1 * 28 * 28)
    input1.foreach(_ => Random.nextDouble())
    val input2 : Array[Double] = Array(1 * 28 * 28)
    input2.foreach(_ => Random.nextDouble())

    val folder = "/home/zhangli/CodeSpace/forTrain"
    val trainData = Paths.get(folder, "/train-images.idx3-ubyte")
    val trainLabel = Paths.get(folder, "/train-labels.idx1-ubyte")
    val validationData = Paths.get(folder, "/t10k-images.idx3-ubyte")
    val validationLabel = Paths.get(folder, "/t10k-labels.idx1-ubyte")
    val batchSize = 2
    val classNum = 10
    val model = LeNet5(classNum)

    val validationSet = DataSet.array(load(validationData, validationLabel))
      .transform(SampleToGreyImg(28, 28))
    val normalizerVal = GreyImgNormalizer(validationSet)
    val valSet = validationSet.transform(normalizerVal)
      .transform(GreyImgToBatch(batchSize))
    val valData = valSet.data(looped = false)

    // init
    val valTrans = new MlTransform()
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
      res = if (res == null) {
        valTrans.transform(testData, params)
      } else {
        res.unionAll(valTrans.transform(testData, params))
      }

      res.printSchema()
      res.show()

      tensorBuffer.clear()
    }
    sc.stop()
  }
}

