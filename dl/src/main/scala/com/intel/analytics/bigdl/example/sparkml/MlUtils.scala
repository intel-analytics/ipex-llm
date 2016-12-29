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
import com.intel.analytics.bigdl.models.alexnet.AlexNet
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.nn.Module
import scopt.OptionParser

object MlUtils {

  val testMean = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
  val testStd = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)
  val imageSize = 32

  sealed trait ModelType

  case object TorchModel extends ModelType

  case object CaffeModel extends ModelType

  case object BigDlModel extends ModelType

  case class PredictParams(
    folder: String = "./",
    partitionNum : Int = 4,
    coreNumber: Int = -1,
    nodeNumber: Int = -1,
    batchSize: Int = 32,
    classNum: Int = 10,

    modelType: ModelType = BigDlModel,
    modelName: String = "",
    caffeDefPath: Option[String] = None,
    modelPath: String = "",
    meanFile: Option[String] = None,
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

    opt[Int]("partitionNum")
      .text("partition num")
      .action((x, c) => c.copy(partitionNum = x))
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

    opt[String]('m', "modelName")
      .text("the model name you want to test")
      .action((x, c) => c.copy(modelName = x.toLowerCase()))

    opt[String]('t', "modelType")
      .text("torch, caffe or bigdl")
      .action((x, c) =>
        x.toLowerCase() match {
          case "torch" => c.copy(modelType = TorchModel)
          case "caffe" => c.copy(modelType = CaffeModel)
          case "bigdl" => c.copy(modelType = BigDlModel)
          case _ =>
            throw new IllegalArgumentException("only torch, caffe or bigdl supported")
        }
      )
    opt[String]("caffeDefPath")
      .text("caffe define path")
      .action((x, c) => c.copy(caffeDefPath = Some(x)))

    opt[String]("meanFile")
      .text("mean file")
      .action((x, c) => c.copy(meanFile = Some(x)))
  }


  def loadModel[@specialized(Float, Double) T](param : PredictParams): Module[Float] = {
    val model = param.modelType match {
      case CaffeModel =>
        param.modelName match {
          case "alexnet" =>
            Module.loadCaffe[Float](AlexNet(param.classNum),
              param.caffeDefPath.get, param.modelPath)
          case "inception" =>
            Module.loadCaffe[Float](Inception_v1_NoAuxClassifier(param.classNum),
              param.caffeDefPath.get, param.modelPath)
        }
      case TorchModel =>
        param.modelName match {
          case "resnet" =>
            Module.loadTorch[Float](param.modelPath)
        }
      case BigDlModel =>
        Module.load[Float](param.modelPath)
      case _ => throw new IllegalArgumentException(s"${param.modelType}")
    }
    model
  }
}
