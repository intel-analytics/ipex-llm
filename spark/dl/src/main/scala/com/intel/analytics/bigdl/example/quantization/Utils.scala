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

package com.intel.analytics.bigdl.example.quantization

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.{ByteRecord, Sample, Transformer}
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy, ValidationMethod, ValidationResult}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scopt.OptionParser
import com.intel.analytics.bigdl.models.lenet.{Utils => LeNetUtils}
import com.intel.analytics.bigdl.models.vgg.{Utils => VggUtils}
import com.intel.analytics.bigdl.nn.Module
import java.io.{File, FileOutputStream, PrintWriter}
import java.nio.file.Paths

object Utils {
  case class TestParams(folder: String = "./",
    model: String = "unkonwn_model",
    modelPath: String = "",
    batchSize: Int = 32)

  val testParser = new OptionParser[TestParams]("BigDL Models Test with Quantization") {
    opt[String]('f', "folder")
            .text("Where's the data location?")
            .action((x, c) => c.copy(folder = x))
    opt[String]("modelPath")
            .text("Where's the model location?")
            .action((x, c) => c.copy(modelPath = x))
            .required()
    opt[String]("model")
            .text("What's the model?")
            .action((x, c) => c.copy(model = x))
            .required()
    opt[Int]('b', "batchSize")
            .text("How many samples in a bach?")
            .action((x, c) => c.copy(batchSize = x))
  }

  def getRddData(model: String, sc: SparkContext, partitionNum: Int,
    folder: String): RDD[ByteRecord] = {
    model match {
      case "lenet" =>
        val validationData = folder + "/t10k-images-idx3-ubyte"
        val validationLabel = folder + "/t10k-labels-idx1-ubyte"
        sc.parallelize(LeNetUtils.load(validationData, validationLabel), partitionNum)

      case "vgg" =>
        sc.parallelize(VggUtils.loadTest(folder), partitionNum)
      case _ => throw new UnsupportedOperationException(s"unknown model: $model")
    }
  }

  def getTransformer(model: String): Transformer[ByteRecord, Sample[Float]] = {
    model match {
      case "lenet" =>
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(LeNetUtils.testMean,
          LeNetUtils.testStd) -> GreyImgToSample()
      case "vgg" =>
        BytesToBGRImg() -> BGRImgNormalizer(VggUtils.testMean, VggUtils.testStd) -> BGRImgToSample()
      case _ => throw new UnsupportedOperationException(s"unknown model: $model")
    }
  }

  def test(model: Module[Float], evaluationSet: RDD[Sample[Float]], batchSize: Int)
  : Array[(ValidationResult, ValidationMethod[Float])] = {
    println(model)
    val result = model.evaluate(evaluationSet, Array(new Top1Accuracy[Float],
      new Top5Accuracy[Float]), Some(batchSize))
    result.foreach(r => println(s"${r._2} is ${r._1}"))
    result
  }

  def testAll(name: String, model: Module[Float], evaluationSet: RDD[Sample[Float]],
    batchSize: Int): Unit = {
    val modelResult = test(model, evaluationSet, batchSize)

    val quantizedModel = Module.quantize(model)
    val quantizedModelResult = test(quantizedModel, evaluationSet, batchSize)

    require(modelResult.length > 0, s"unknown result")
    val method = modelResult(0)._2.toString()
    val totalNum = modelResult(0)._1.result()._2

    val accuracies = new Array[(Float, Float)](modelResult.length)
    modelResult.indices.foreach { i =>
      val accuracy = (modelResult(i)._1.result()._1, quantizedModelResult(i)._1.result()._1)
      accuracies(i) = accuracy
    }

    def writeToLog(model: String, totalNum: Int, accuracies: Array[(Float, Float)]): Unit = {
      val name = Paths.get(System.getProperty("user.dir"), "model_inference.log").toString
      val file = new File(name)

      val out = if (file.exists() && !file.isDirectory) {
        new PrintWriter(new FileOutputStream(new File(name), true))
      } else {
        new PrintWriter(name)
      }

      out.append(model)
      out.append("\t" + totalNum.toString)
      accuracies.foreach(a => out.append(s"\t${a._1}-${a._2}"))
      out.append("\n")
      out.close()
    }

    writeToLog(name, totalNum, accuracies)
  }
}
