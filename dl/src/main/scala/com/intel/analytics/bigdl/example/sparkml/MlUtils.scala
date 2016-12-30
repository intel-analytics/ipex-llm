/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.example.sparkml

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.image.{BGRImage, LocalLabeledImagePath}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, Row}
import scopt.OptionParser

import scala.reflect.ClassTag

object MlUtils {

  val testMean = (0.485, 0.456, 0.406)
  val testStd = (0.229, 0.224, 0.225)

  val imageSize = 224

  sealed trait ModelType

  case object TorchModel extends ModelType

  case object CaffeModel extends ModelType

  case object BigDlModel extends ModelType

  case class PredictParams(
    folder: String = "./",
    coreNumber: Int = -1,
    nodeNumber: Int = -1,
    batchSize: Int = 32,
    classNum: Int = 1000,

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

    opt[Int]('c', "core")
      .text("cores number on each node")
      .action((x, c) => c.copy(coreNumber = x))

    opt[Int]('n', "nodeNumber")
      .text("nodes number to train the model")
      .action((x, c) => c.copy(nodeNumber = x))

    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))

    opt[Int]("classNum")
      .text("class num")
      .action((x, c) => c.copy(classNum = x))

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

  case class DfPoint(features: DenseVector, imageName: String)

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
}
