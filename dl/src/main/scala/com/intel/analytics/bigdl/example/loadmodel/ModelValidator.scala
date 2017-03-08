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

package com.intel.analytics.bigdl.example.loadmodel

import java.nio.file.Paths

import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy, Validator}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.language.existentials

/**
 * ModelValidator provides an integrated example to load models,
 * and test over imagenet validation dataset
 * (running as a local Java program, or a standard Spark program).
 */
object ModelValidator {

  val logger = Logger.getLogger(getClass)

  sealed trait ModelType

  case object TorchModel extends ModelType

  case object CaffeModel extends ModelType

  case object BigDlModel extends ModelType

  case class TestLocalParams(
    folder: String = "./",
    modelType: ModelType = null,
    modelName: String = "",
    caffeDefPath: Option[String] = None,
    modelPath: String = "",
    batchSize: Int = 32,
    meanFile: Option[String] = None,
    coreNumber: Int = Runtime.getRuntime().availableProcessors() / 2,
    nodeNumber: Int = -1,
    env: String = "local"
  )

  val testLocalParser = new OptionParser[TestLocalParams]("BigDL Image Classifier Example") {
    head("BigDL Image Classifier Example")
    opt[String]('f', "folder")
      .text("where you put your local image files")
      .action((x, c) => c.copy(folder = x))
    opt[String]('m', "modelName")
      .text("the model name you want to test")
      .required()
      .action((x, c) => c.copy(modelName = x.toLowerCase()))
    opt[String]('t', "modelType")
      .text("torch, caffe or bigdl")
      .required()
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
    opt[String]("modelPath")
      .text("model path")
      .action((x, c) => c.copy(modelPath = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[String]("meanFile")
      .text("mean file")
      .action((x, c) => c.copy(meanFile = Some(x)))
    opt[Int]('c', "core")
      .text("cores number to test the model")
      .action((x, c) => c.copy(coreNumber = x))
    opt[Int]('n', "node")
      .text("node number to test the model")
      .action((x, c) => c.copy(nodeNumber = x))
    opt[String]("env")
      .text("execution environment")
      .validate(x => {
        if (Set("local", "spark").contains(x.toLowerCase)) {
          success
        } else {
          failure("env only support local|spark")
        }
      })
      .action((x, c) => c.copy(env = x.toLowerCase()))
      .required()
  }

  def main(args: Array[String]): Unit = {
    testLocalParser.parse(args, TestLocalParams()).foreach(param => {
      Engine.setCoreNumber(param.coreNumber)
      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark")
        .map(conf => {
          conf.setAppName("BigDL Image Classifier Example")
            .set("spark.akka.frameSize", 64.toString)
          new SparkContext(conf)
        })

      val valPath = Paths.get(param.folder, "val")

      val (model, validateDataSet) = param.modelType match {
        case CaffeModel =>
          param.modelName match {
            case "alexnet" =>
              (Module.loadCaffe[Float](AlexNet(1000),
                param.caffeDefPath.get, param.modelPath),
                AlexNetPreprocessor(valPath, param.batchSize, param.meanFile.get, sc))
            case "inception" =>
              (Module.loadCaffe[Float](Inception_v1_NoAuxClassifier(1000),
                param.caffeDefPath.get, param.modelPath),
                InceptionPreprocessor(valPath, param.batchSize, sc))
          }

        case TorchModel =>
          param.modelName match {
            case "resnet" =>
              (Module.loadTorch[Float](param.modelPath),
                ResNetPreprocessor(valPath, param.batchSize, sc))
          }

        case _ => throw new IllegalArgumentException(s"${ param.modelType } is not" +
          s"supported in this example, please use alexnet/inception/resnet")
      }
      println(model)

      val validator = Validator(model, validateDataSet)
      val evaluator = Array(new Top1Accuracy[Float](), new Top5Accuracy[Float]())
      val result = validator.test(evaluator)
      result.foreach(r => {
        logger.info(s"${ r._2 } is ${ r._1 }")
      })
    })
  }
}
