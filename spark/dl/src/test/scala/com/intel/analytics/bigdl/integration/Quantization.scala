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
package com.intel.analytics.bigdl.integration

import java.nio.file.Paths

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.{ByteRecord, Sample, Transformer}
import com.intel.analytics.bigdl.models.lenet.{Utils => LeNetUtils}
import com.intel.analytics.bigdl.models.resnet.{Utils => ResNetUtils}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Integration
class QuantizationSpec extends FlatSpec with Matchers with BeforeAndAfter{
  def test(model: Module[Float], evaluationSet: RDD[Sample[Float]], batchSize: Int)
  : Array[(ValidationResult, ValidationMethod[Float])] = {
    println(model)
    val result = model.evaluate(evaluationSet, Array(new Top1Accuracy[Float],
      new Top5Accuracy[Float]), Some(batchSize))
    result.foreach(r => println(s"${r._2} is ${r._1}"))
    result
  }

  type Result[Float] = Array[(ValidationResult, ValidationMethod[Float])]
  def checkResult(fp32: Result[Float], int8: Result[Float]): Unit = {
    fp32.zip(int8).foreach{ r =>
      val a1 = r._1._1.result()._1
      val a2 = r._2._1.result()._1
      require(Math.abs(a1 - a2) < 0.01, s"accuracy of quantized model seems wrong")
    }
  }

  def getRddData(model: String, sc: SparkContext, partitionNum: Int,
    folder: String): RDD[ByteRecord] = {
    model match {
      case "lenet" =>
        val validationData = folder + "/t10k-images-idx3-ubyte"
        val validationLabel = folder + "/t10k-labels-idx1-ubyte"
        sc.parallelize(LeNetUtils.load(validationData, validationLabel), partitionNum)

      case "resnet" =>
        sc.parallelize(ResNetUtils.loadTest(folder), partitionNum)

      case _ => throw new UnsupportedOperationException(s"unknown model: $model")
    }
  }

  def getTransformer(model: String): Transformer[ByteRecord, Sample[Float]] = {
    model match {
      case "lenet" =>
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(LeNetUtils.testMean,
          LeNetUtils.testStd) -> GreyImgToSample()

      case "resnet" =>
        import com.intel.analytics.bigdl.models.resnet.Cifar10DataSet

        BytesToBGRImg() -> BGRImgNormalizer(Cifar10DataSet.trainMean,
          Cifar10DataSet.trainStd) -> BGRImgToSample()

      case _ => throw new UnsupportedOperationException(s"unknown model: $model")
    }
  }

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  "Quantize LeNet5" should "generate the same top1 accuracy" in {
    val lenetFP32Model = System.getenv("lenetfp32model")
    val mnist = System.getenv("mnist")

    val conf = Engine.createSparkConf()
            .setAppName(s"Test LeNet5 with quantization")
            .set("spark.akka.frameSize", 64.toString)
            .setMaster("local[4]")
    val sc = new SparkContext(conf)
    Engine.init

    val partitionNum = Engine.nodeNumber() * Engine.coreNumber()
    val rddData = getRddData("lenet", sc, partitionNum, mnist)
    val transformer = getTransformer("lenet")
    val evaluationSet = transformer(rddData)

    val batchSize = Engine.coreNumber() * Engine.nodeNumber() * 4

    val model = Module.loadModule(lenetFP32Model)
    val fp32Result = test(model, evaluationSet, batchSize)

    val int8Model = model.quantize()
    val int8Result = test(int8Model, evaluationSet, batchSize)

    checkResult(fp32Result, int8Result)

    val tempDir = Paths.get(System.getProperty("java.io.tmpdir"))
    val modelPath = Paths.get(tempDir.toString, "lenet.quantized.bigdlmodel")
    int8Model.saveModule(modelPath.toString, overWrite = true)
    sc.stop()
  }

  "Quantize ResNet on Cifar" should "generate the same top1 accuracy" in {
    val resnetFP32Model = System.getenv("resnetfp32model")
    val cifar10 = System.getenv("cifar10")

    val conf = Engine.createSparkConf()
            .setAppName(s"Test ResNet on Cifar10 with quantization")
            .set("spark.akka.frameSize", 64.toString)
            .setMaster("local[4]")
    val sc = new SparkContext(conf)
    Engine.init

    val partitionNum = Engine.nodeNumber() * Engine.coreNumber()
    val batchSize = Engine.coreNumber() * Engine.nodeNumber() * 4

    val rddData = getRddData("resnet", sc, partitionNum, cifar10)
    val transformer = getTransformer("resnet")
    val evaluationSet = transformer(rddData)

    val model = Module.loadModule(resnetFP32Model)
    val fp32Result = test(model, evaluationSet, batchSize)

    val int8Model = model.quantize()
    val int8Result = test(int8Model, evaluationSet, batchSize)

    checkResult(fp32Result, int8Result)
    sc.stop()
  }

  "Load quantized model of LeNet5 on mnist" should "generate the same top1 accuracy" in {
    val lenetFP32Model = System.getenv("lenetfp32model")
    val mnist = System.getenv("mnist")

    val tempDir = Paths.get(System.getProperty("java.io.tmpdir"))
    val modelPath = Paths.get(tempDir.toString, "lenet.quantized.bigdlmodel")
    val lenetInt8Model = modelPath.toString


    val conf = Engine.createSparkConf()
            .setAppName(s"Test LeNet5 with quantization")
            .set("spark.akka.frameSize", 64.toString)
            .setMaster("local[4]")
    val sc = new SparkContext(conf)
    Engine.init

    val partitionNum = Engine.nodeNumber() * Engine.coreNumber()
    val rddData = getRddData("lenet", sc, partitionNum, mnist)
    val transformer = getTransformer("lenet")
    val evaluationSet = transformer(rddData)

    val batchSize = Engine.coreNumber() * Engine.nodeNumber() * 4

    val model = Module.loadModule(lenetFP32Model)
    val fp32Result = test(model, evaluationSet, batchSize)

    val int8Model = Module.loadModule(lenetInt8Model)
    val int8Result = test(int8Model, evaluationSet, batchSize)

    checkResult(fp32Result, int8Result)
    sc.stop()
  }
}
