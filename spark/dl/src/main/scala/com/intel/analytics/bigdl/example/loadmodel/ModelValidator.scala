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

  /**
   * This is a trait meaning the model type.
   * There are three sorts of model type, which
   * are torch model [[TorchModel]], caffe model
   * [[CaffeModel]] and BigDL model [[BigDlModel]].
   */
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
    meanFile: Option[String] = None
  )

  val testLocalParser = new OptionParser[TestLocalParams]("BigDL Load Model Example") {
    head("BigDL Load Model Example")
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
  }

  def main(args: Array[String]): Unit = {
    testLocalParser.parse(args, TestLocalParams()).foreach(param => {
      val conf = Engine.createSparkConf()
      conf.setAppName("BigDL Image Classifier Example")
      val sc = new SparkContext(conf)
      Engine.init

      val valPath = param.folder

      val (model, validateDataSet) = param.modelType match {
        case CaffeModel =>
          param.modelName match {
            case "alexnet" =>
              (Module.loadCaffeModel[Float](
                param.caffeDefPath.get, param.modelPath),
                AlexNetPreprocessor.rdd(valPath, param.batchSize, param.meanFile.get, sc))
            case "inception" =>
              (Module.loadCaffeModel[Float](
                param.caffeDefPath.get, param.modelPath),
                InceptionPreprocessor.rdd(valPath, param.batchSize, sc))
          }

        case TorchModel =>
          param.modelName match {
            case "resnet" =>
              (Module.loadTorch[Float](param.modelPath),
                ResNetPreprocessor.rdd(valPath, param.batchSize, sc))
          }

        case BigDlModel =>
          param.modelName match {
            case "resnet" =>
              (Module.loadModule[Float](param.modelPath),
                ResNetPreprocessor.rdd(valPath, param.batchSize, sc, BigDlModel))
            case "vgg16" =>
              (Module.loadModule[Float](param.modelPath),
                VGGPreprocessor.rdd(valPath, param.batchSize, sc))
          }

        case _ => throw new IllegalArgumentException(s"${ param.modelType } is not" +
          s"supported in this example, please use alexnet/inception/resnet")
      }
      println(model)

      val result = model.evaluate(
        validateDataSet,
        Array(new Top1Accuracy[Float](), new Top5Accuracy[Float]()),
        Some(param.batchSize))

      result.foreach(r => {
        logger.info(s"${ r._2 } is ${ r._1 }")
      })
      sc.stop()
    })
  }
}
