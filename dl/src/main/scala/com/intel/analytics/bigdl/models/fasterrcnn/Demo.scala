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

package com.intel.analytics.bigdl.models.fasterrcnn

import com.intel.analytics.bigdl.models.fasterrcnn.model.Model._
import com.intel.analytics.bigdl.models.fasterrcnn.model._
import com.intel.analytics.bigdl.models.fasterrcnn.tools.{FrcnnDistriValidator, FrcnnLocalValidator, Preprocessor}
import com.intel.analytics.bigdl.models.fasterrcnn.utils.VisualizeTool
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

object Demo {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.models.fasterrcnn").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  val classes = Array[String](
    "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
  )

  case class PascolVocDemoParam(imageFolder: String = "",
    outputFolder: String = "data/demo",
    modelType: ModelType = Model.PVANET,
    caffeDefPath: String = "",
    caffeModelPath: String = "",
    coreNumber: Int = -1,
    nodeNumber: Int = -1,
    env: String = "local")

  val parser = new OptionParser[PascolVocDemoParam]("Spark-DL Faster-RCNN Demo") {
    head("Spark-DL Faster-RCNN Demo")
    opt[String]('f', "folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(imageFolder = x))
      .required()
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
      .required()
    opt[String]('t', "modelType")
      .text("net type : VGG16 | PVANET")
      .action((x, c) => c.copy(modelType = Model.withName(x)))
      .required()
    opt[String]("caffeDefPath")
      .text("caffe prototxt")
      .action((x, c) => c.copy(caffeDefPath = x))
      .required()
    opt[String]("caffeModelPath")
      .text("caffe model path")
      .action((x, c) => c.copy(caffeModelPath = x))
      .required()
    opt[Int]('c', "core")
      .text("cores number on each node")
      .action((x, c) => c.copy(coreNumber = x))
      .required()
      .required()
    opt[Int]('n', "nodeNumber")
      .text("nodes number to train the model")
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
    val params = parser.parse(args, PascolVocDemoParam()).get
    if (params.env == "local") {
      Engine.setCoreNumberFixed(params.coreNumber)
    }
    val sc = Engine.init(params.nodeNumber, params.coreNumber, params.env == "spark")
      .map(conf => {
        conf.setAppName("Spark-DL Faster-RCNN Demo")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1")
        new SparkContext(conf)
      })

    val net = FasterRcnn(params.modelType,
      caffeModel = Some((params.caffeDefPath, params.caffeModelPath)))

    val output = if (sc.isDefined) {
      val rdd = Preprocessor.processSeqFileDistri(params.nodeNumber,
        params.coreNumber, params.imageFolder, net.param, sc.get)
      val validator = new FrcnnDistriValidator(net, classNum = 21, rdd = rdd,
        maxPerImage = -1, thresh = 0)
      validator.test()
    } else {
      val dataSet = Preprocessor.processSeqFileLocal(params.imageFolder, net.param)
      val validator = new FrcnnLocalValidator(net, classNum = 21, dataSet = dataSet,
        maxPerImage = -1, thresh = 0)
      validator.test()
    }
    var i = 0
    var cls = 1
    while (i < output.length) {
      cls = 1
      while (cls < classes.length) {
        VisualizeTool.visDetection(output(i)._4, classes(cls), output(i)._1(cls),
          thresh = 0.8f, outPath = params.outputFolder)
        cls += 1
      }
      i += 1
    }
    logger.info(s"labeled images are saved to ${params.outputFolder}")
  }
}
