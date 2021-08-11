/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.example.imageclassification

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.image.{BGRImage, LocalLabeledImagePath}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dataset.DataSet.SeqFileFolder
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import scopt.OptionParser

import scala.reflect.ClassTag

object MlUtils {

  val testMean = (0.485, 0.456, 0.406)
  val testStd = (0.229, 0.224, 0.225)

  val imageSize = 224

  /**
   * This is a trait meaning the model type.
   * There are two sorts of model type, which
   * are torch model [[TorchModel]] and BigDL
   * model [[BigDlModel]].
   */
  sealed trait ModelType

  case object TorchModel extends ModelType

  case object BigDlModel extends ModelType

  case class PredictParams(
    folder: String = "./",
    batchSize: Int = 32,
    classNum: Int = 1000,
    isHdfs: Boolean = false,

    modelType: ModelType = BigDlModel,
    modelPath: String = "",
    showNum : Int = 100
  )

  val predictParser = new OptionParser[PredictParams]("BigDL Predict Example") {
    opt[String]('f', "folder")
      .text("where you put the test data")
      .action((x, c) => c.copy(folder = x))
      .required()

    opt[String]("modelPath")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelPath = x))
      .required()

    opt[Int]('b', "batchSizePerCore")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))

    opt[Int]("classNum")
      .text("class num")
      .action((x, c) => c.copy(classNum = x))

    opt[Boolean]("isHdfs")
      .text("whether the input data is from Hdfs or not")
      .action((x, c) => c.copy(isHdfs = x))

    opt[Int]("showNum")
      .text("show num")
      .action((x, c) => c.copy(showNum = x))

    opt[String]('f', "folder")
      .text("where you put your local image files")
      .action((x, c) => c.copy(folder = x))

    opt[String]('t', "modelType")
      .text("torch, bigdl")
      .action((x, c) =>
        x.toLowerCase() match {
          case "torch" => c.copy(modelType = TorchModel)
          case "bigdl" => c.copy(modelType = BigDlModel)
          case _ =>
            throw new IllegalArgumentException("only torch, bigdl supported")
        }
      )
  }

  def loadModel[@specialized(Float, Double) T : ClassTag](param : PredictParams)
    (implicit ev: TensorNumeric[T]): Module[T] = {
    val model = param.modelType match {
      case TorchModel =>
        Module.loadTorch[T](param.modelPath)
      case BigDlModel =>
        Module.load[T](param.modelPath)
      case _ => throw new IllegalArgumentException(s"${param.modelType}")
    }
    model
  }

  /**
   * It is used to store single data frame information
   *
   * @param features extracted features after the transformers
   * @param imageName image name
   */
  case class DfPoint(features: DenseVector, imageName: String)

  /**
   * [[ByteImage]] is case class, which represents an object
   * of image in byte format.
   *
   * @param data image byte data
   * @param imageName image name
   */
  case class ByteImage(data: Array[Byte], imageName: String)

  def transformDF(data: DataFrame, f: Transformer[Row, DenseVector]): DataFrame = {
    val vectorRdd = data.select("data").rdd.mapPartitions(f(_))
    val dataRDD = data.rdd.zipPartitions(vectorRdd) { (a, b) =>
      b.zip(a.map(_.getAs[String]("imageName")))
        .map(
        v => DfPoint(v._1, v._2)
      )
    }
    data.sqlContext.createDataFrame(dataRDD)
  }

  def imagesLoad(paths: Array[LocalLabeledImagePath], scaleTo: Int):
    Array[ByteImage] = {
    var count = 1
    val buffer = paths.map(imageFile => {
      count += 1
      ByteImage(BGRImage.readImage(imageFile.path, scaleTo), imageFile.path.getFileName.toString)
    })
    buffer
  }

  def imagesLoadSeq(url: String, sc: SparkContext, classNum: Int): RDD[ByteImage] = {
    sc.sequenceFile(url, classOf[Text], classOf[Text]).map(image => {
      ByteImage(image._2.copyBytes(), SeqFileFolder.readName(image._1))
    })
  }
}
