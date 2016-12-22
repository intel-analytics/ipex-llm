/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.example.loadCaffe

import java.nio.file.Paths

import com.intel.analytics.bigdl.models.alexnet.AlexNet
import com.intel.analytics.bigdl.models.googlenet.GoogleNet_v1_NoAuxClassifier
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{LocalValidator, Top1Accuracy, Top5Accuracy}
import org.apache.log4j.Logger
import scopt.OptionParser


object Test {

  val logger = Logger.getLogger(getClass)

  case class TestLocalParams(
    folder: String = "./",
    modelType: String = "alexnet",
    caffeDefPath: String = "",
    caffeModelPath: String = "",
    batchSize: Int = 32,
    meanFile: Option[String] = None
  )

  val testLocalParser = new OptionParser[TestLocalParams]("BigDL LoadCaffe Example") {
    head("LoadCaffe example")
    opt[String]('f', "folder")
      .text("where you put your local hadoop sequence files")
      .action((x, c) => c.copy(folder = x))
    opt[String]('m', "modelType")
      .text("the type of model you want to test")
      .action((x, c) => c.copy(modelType = x))
    opt[String]("caffeDefPath")
      .text("caffe define path")
      .action((x, c) => c.copy(caffeDefPath = x))
    opt[String]("caffeModelPath")
      .text("caffe model path")
      .action((x, c) => c.copy(caffeModelPath = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[String]('m', "meanFile")
      .text("mean file")
      .action((x, c) => c.copy(meanFile = Some(x)))
  }

  def main(args: Array[String]): Unit = {
    testLocalParser.parse(args, TestLocalParams()).foreach(param => {
      val valPath = Paths.get(param.folder, "val")
      val imageSize = param.modelType match {
        case "alexnet" => 227
        case "googlenet" => 224
      }

      val (module, validateDataSet) = param.modelType match {
        case "alexnet" =>
          (AlexNet(1000), AlexNetPreprocessor(valPath, imageSize,
            param.batchSize, param.meanFile.get))
        case "googlenet" =>
          (GoogleNet_v1_NoAuxClassifier(1000),
            GoogleNetPreprocessor(valPath, imageSize, param.batchSize))
      }

      val model = Module.loadCaffeParameters[Float](module,
        param.caffeDefPath, param.caffeModelPath)
      model.evaluate()
      val validator = new LocalValidator[Float](model)
      val result = validator.test(validateDataSet, Array(new Top1Accuracy[Float],
        new Top5Accuracy[Float]))
      result.foreach(r => {
        logger.info(s"${r._2} is ${r._1}")
      })
    })
  }
}
