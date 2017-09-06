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

package com.intel.analytics.bigdl.example.bigquant

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.{ByteRecord, DataSet, Sample, Transformer}
import com.intel.analytics.bigdl.models.lenet.{Utils => LeNetUtils}
import com.intel.analytics.bigdl.models.vgg.{Utils => VggUtils}
import com.intel.analytics.bigdl.models.resnet.{Cifar10DataSet => ResNetCifar10DataSet, Utils => ResNetUtils}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.caffe.CaffeLoader
import java.io.{File, FileOutputStream, PrintWriter}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scopt.OptionParser

object Utils {
  case class TestParams(folder: String = "./",
    model: String = "unkonwn_model",
    modelPath: String = "",
    batchSize: Int = 32,
    quantize: Boolean = true)

  val testParser = new OptionParser[TestParams]("BigDL Models Test with Quant") {
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
    opt[Boolean]('q', "quantize")
            .text("Quantize the model?")
            .action((x, c) => c.copy(quantize = x))
  }

  def getRddData(model: String, sc: SparkContext, partitionNum: Int,
    folder: String): RDD[ByteRecord] = {
    def imagenet: RDD[ByteRecord] = DataSet.SeqFileFolder.filesToRdd(folder, sc, 1000)
    model match {
      case "lenet" =>
        val validationData = folder + "/t10k-images-idx3-ubyte"
        val validationLabel = folder + "/t10k-labels-idx1-ubyte"
        sc.parallelize(LeNetUtils.load(validationData, validationLabel), partitionNum)

      case "vgg" =>
        sc.parallelize(VggUtils.loadTest(folder), partitionNum)

      case m if m.contains("alexnet") => imagenet
      case m if m.contains("inception_v1") || m.contains("googlenet") => imagenet
      case "inception_v2" => imagenet
      case m if m.toLowerCase.contains("resnet") && !m.toLowerCase.contains("cifar10") => imagenet
      case m if m.toLowerCase.contains("resnet") && m.toLowerCase.contains("cifar10") =>
        sc.parallelize(ResNetUtils.loadTest(folder), partitionNum)

      case _ => throw new UnsupportedOperationException(s"unknown model: $model")
    }
  }

  def getTransformer(model: String): Transformer[ByteRecord, Sample[Float]] = {
    def imagenetPreprocessing(imageSize: Int): Transformer[ByteRecord, Sample[Float]] = {
      val name = Paths.get(System.getProperty("user.dir"), "mean.txt").toString
      val means = loadMeanFile(name)
      BytesToBGRImg(normalize = 1f) -> BGRImgCropper(256, 256, CropCenter) ->
              BGRImgPixelNormalizer(means) -> BGRImgCropper(imageSize, imageSize, CropCenter) ->
              BGRImgToSample(toRGB = false)
    }

    model match {
      case "lenet" =>
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(LeNetUtils.testMean,
          LeNetUtils.testStd) -> GreyImgToSample()

      case "vgg" =>
        BytesToBGRImg() -> BGRImgNormalizer(VggUtils.testMean, VggUtils.testStd) -> BGRImgToSample()

      case m if m.contains("alexnet") => imagenetPreprocessing(227)

      case m if m.contains("inception_v1") || m.contains("googlenet") =>
        BytesToBGRImg(normalize = 1f) ->
                BGRImgCropper(224, 224, CropCenter) ->
                BGRImgNormalizer(123, 117, 104, 1, 1, 1) -> BGRImgToSample(toRGB = false)

      case "inception_v2" =>
        BytesToBGRImg() -> BGRImgCropper(224, 224, CropCenter) ->
                HFlip(0.5) -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) ->
                BGRImgToSample()

      case m if m.toLowerCase.contains("resnet") && !m.toLowerCase.contains("cifar10") =>
        BytesToBGRImg() -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) ->
                BGRImgCropper(224, 224, CropCenter) -> BGRImgToSample()
      case m if m.toLowerCase.contains("resnet") && m.toLowerCase.contains("cifar10") =>
        BytesToBGRImg() -> BGRImgNormalizer(ResNetCifar10DataSet.trainMean,
          ResNetCifar10DataSet.trainStd) -> BGRImgToSample()

      case _ => throw new UnsupportedOperationException(s"unknown model: $model")
    }
  }

  def time[R](block: => R): (R, Double) = {
    val start = System.nanoTime()
    val result = block
    val end = System.nanoTime()
    (result, (end - start) / 1e9)
  }

  def test(model: Module[Float], evaluationSet: RDD[Sample[Float]], batchSize: Int)
  : Array[(ValidationResult, ValidationMethod[Float])] = {
    println(model)
    val result = model.evaluate(evaluationSet, Array(new Top1Accuracy[Float],
      new Top5Accuracy[Float]), Some(batchSize))
    result.foreach(r => println(s"${r._2} is ${r._1}"))
    result
  }

  def writeToLog(model: String, quantized: Boolean, totalNum: Int, accuracies: Array[Float],
    costs: Double): Unit = {
    val name = Paths.get(System.getProperty("user.dir"), "model_inference.log").toString
    val file = new File(name)

    val out = if (file.exists() && !file.isDirectory) {
      new PrintWriter(new FileOutputStream(new File(name), true))
    } else {
      new PrintWriter(name)
    }

    out.append(model)
    if (quantized) {
      out.append("\tQuantized")
    } else {
      out.append("\tMKL")
    }
    out.append("\t" + totalNum.toString)
    accuracies.foreach(a => out.append(s"\t${a}"))
    out.append(s"\t${costs}")
    out.append("\n")
    out.close()
  }

  def writeToLog(model: String, totalNum: Int, accuracies: Array[(Float, Float)],
    costs: List[Double]): Unit = {
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
    out.append(s"\t${costs.mkString("-")}")
    out.append("\n")
    out.close()
  }

  def testAll(name: String, model: Module[Float], evaluationSet: RDD[Sample[Float]],
    batchSize: Int): Unit = {
    val (modelResult, modelCosts) = time {
      test(model, evaluationSet, batchSize)
    }

    val quantizedModel = model.quantize()
    val (quantizedModelResult, quantizedModelCosts) = time {
      test(quantizedModel, evaluationSet, batchSize)
    }

    require(modelResult.length > 0, s"unknown result")
    val totalNum = modelResult(0)._1.result()._2

    val accuracies = new Array[(Float, Float)](modelResult.length)
    modelResult.indices.foreach { i =>
      val accuracy = (modelResult(i)._1.result()._1, quantizedModelResult(i)._1.result()._1)
      accuracies(i) = accuracy
    }

    val costs = List(modelCosts, quantizedModelCosts).map { x =>
      Math.round(totalNum / x * 100) / 100.0
    }

    writeToLog(name, totalNum, accuracies, costs)
  }

  def convertModelFromCaffe(prototxt: String, caffeModel: String): Module[Float] = {
    CaffeLoader.loadCaffe[Float](prototxt, caffeModel)._1
  }

  def loadMeanFile(path: String): Tensor[Float] = {
    val lines = Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8)
    val array = new Array[Float](lines.size())

    lines.toArray.zipWithIndex.foreach {x =>
      array(x._2) = x._1.toString.toFloat
    }

    Tensor[Float](array, Array(array.length))
  }
}
