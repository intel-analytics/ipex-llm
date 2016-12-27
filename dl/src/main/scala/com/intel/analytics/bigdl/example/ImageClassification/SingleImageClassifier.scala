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

package com.intel.analytics.bigdl.example.ImageClassification

import java.nio.file.{Path, Paths}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.example.ImageClassification.ImageClassifier._
import com.intel.analytics.bigdl.models.alexnet.AlexNet
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.Logger
import scopt.OptionParser

import scala.io.Source
import scala.util.matching.Regex

class SingleImageClassifier(model: Module[Float], classNameMappingFile: String) {

  val logger = Logger.getLogger(getClass)
  var classIdToString: Array[String] = loadClassIdToStringMapping()
  model.evaluate()

  def loadClassIdToStringMapping(): Array[String] = {
    val pattern = new Regex("""(n\d*) ([\S, ]*)""", "class", "name")
    Source.fromFile(classNameMappingFile)
      .getLines().map(line => {
      val result = pattern.findFirstMatchIn(line).get
      (result.group("class"), result.group("name"))
    }).toArray.sortBy(_._1).map(x => x._2)
  }

  def inferenceOnImage(dataSet: LocalDataSet[MiniBatch[Float]], numTopPrediction: Int): Unit = {
    val dataIter = dataSet.data(train = false)
    while (dataIter.hasNext) {
      val data = dataIter.next()
      val output = model.forward(data.data).asInstanceOf[Tensor[Float]].squeeze()
        .apply1(x => Math.exp(x).toFloat)
      val resultSize = Math.min(numTopPrediction, output.nElement())
      val topResults = output.topk(resultSize, 1, increase = false)
      logger.info(s"Top $numTopPrediction results: -----------")
      for (i <- 1 to resultSize) {
        val id = topResults._2.valueAt(i).toInt
        val humanString = classIdToString(id - 1)
        logger.info("%s (score = %.5f)".format(humanString, topResults._1.valueAt(i)))
      }
    }
  }
}


object SingleImageClassifier {

  case class SinaleImageLocalParams(
    modelType: ModelType = null,
    modelName: String = "",
    caffeDefPath: Option[String] = None,
    modelPath: String = "",
    meanFile: Option[String] = None,
    imagePath: Path = Paths.get("/tmp/imagenet/cat.jpg"),
    numTopPrediction: Int = 5,
    classNameMapFile: String = "/tmp/imagenet/synset_words.txt"
  )

  val testLocalParser = new OptionParser[SinaleImageLocalParams]("SingleImageClassifier Example") {
    head("Simple Image Classifier Example")
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
    opt[String]('m', "meanFile")
      .text("mean file")
      .action((x, c) => c.copy(meanFile = Some(x)))
    opt[String]('i', "imagePath")
      .text("single image file path")
      .action((x, c) => c.copy(imagePath = Paths.get(x)))
    opt[Int]('n', "numTopPrediction")
      .text("display this many predictions")
      .action((x, c) => c.copy(numTopPrediction = x))
  }

  def main(args: Array[String]): Unit = {

    testLocalParser.parse(args, SinaleImageLocalParams()).foreach(param => {
      val (model, data) = param.modelType match {
        case CaffeModel =>
          param.modelName match {
            case "alexnet" =>
              (Module.loadCaffe[Float](AlexNet(1000),
                param.caffeDefPath.get, param.modelPath),
                AlexNetPreprocessor(param.imagePath, 1, param.meanFile.get, hasLabel = false))
            case "inception" =>
              (Module.loadCaffe[Float](Inception_v1_NoAuxClassifier(1000),
                param.caffeDefPath.get, param.modelPath),
                InceptionPreprocessor(param.imagePath, 1, hasLabel = false))
          }

        case TorchModel =>
          param.modelName match {
            case "resnet" =>
              (Module.loadTorch[Float](param.modelPath),
                ResNetPreprocessor(param.imagePath, 1, hasLabel = false))
          }

        case _ => throw new IllegalArgumentException(s"${param.modelType}")
      }
      val classifier = new SingleImageClassifier(model, param.classNameMapFile)
      classifier.inferenceOnImage(data.toLocal(), param.numTopPrediction)
    })
  }
}
