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

package com.intel.analytics.sparkdl.example

import java.awt.color.ColorSpace
import java.util

import com.intel.analytics.sparkdl.models.imagenet.ResNet
import com.intel.analytics.sparkdl.models.imagenet.ResNet.ShortcutType
import com.intel.analytics.sparkdl.nn.{CrossEntropyCriterion, Module, SpatialBatchNormalization, SpatialConvolution}
import com.intel.analytics.sparkdl.optim.{EvaluateMethods, SGD}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.{File, T}

import scala.collection.mutable
import scala.reflect.ClassTag

object ImageNetLocal {
  val startTime = System.nanoTime()

  val regimes = Map(
    "alexnet" -> Array(
      (1, 18, 1e-2, 5e-4),
      (19, 29, 5e-3, 5e-4),
      (30, 43, 1e-3, 0.0),
      (44, 52, 5e-4, 0.0),
      (53, 100000000, 1e-4, 0.0)
    ),
    "googlenet-cf" -> Array(
      (1, 18, 1e-2, 2e-4),
      (19, 29, 5e-3, 2e-4),
      (30, 43, 1e-3, 0.0),
      (44, 52, 5e-4, 0.0),
      (53, 100000000, 1e-4, 0.0)
    )
  )

  def log(msg: String): Unit = {
    println(s"[${(System.nanoTime() - startTime) / 1e9}s] $msg")
  }

  def run(donkey: Donkey, dataSet: DataSets, netType: String, classNum: Int,
    labelsMap: Map[String, Double], testInterval: Int, donkeyVal: Donkey,
    dataSetVal: DataSets, batchSize: Int, modelPath : String, modelDepth: Int): Unit = {
    // Compute Mean on amount of samples
    //val samples = 10000
    /*val samples = 100
    log(s"Start to calculate Mean on $samples samples")
    var (meanR, meanG, meanB) = Array.tabulate(samples)(n => {
      //print(".")
      val data = donkey.pull
      dataSet.post(data._2)
      ImageNetUtils.computeMean(data._1, data._2.dataOffset)
    }).reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
    meanR /= samples
    meanG /= samples
    meanB /= samples
    println()

    // Compute std on amount of samples
    log(s"Start to calculate std on $samples samples")
    var (varR, varG, varB) = Array.tabulate(samples)(n => {
      //print(".")
      val data = donkey.pull
      dataSet.post(data._2)
      ImageNetUtils.computeVar(data._1, meanR, meanG, meanB, data._2.dataOffset)
    }).reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
    varR /= samples
    varG /= samples
    varB /= samples*/

    val model = netType match {
      case "alexnet" => AlexNet.getModel[Float](classNum)
      case "googlenet" => GoogleNet.getModel[Float](classNum)
      case "googlenet-bn" => GoogleNet.getModel[Float](classNum, "googlenet-bn")
      case "googlenet-cf" => GoogleNet.getModelCaffe[Float](classNum)
      case "resnet" => ResNet[Float](classNum, T("shortcutType" -> ShortcutType.B, "depth" -> modelDepth))
      case _ => throw new IllegalArgumentException
    }
    if (netType == "resnet") {
      shareGradInput(model)
      convInit("SpatialConvolution", model)
      bnInit("SpatialBatchNormalization", model)
    }


    val (weights, grad) = model.getParameters()
    println(s"modelsize ${weights.nElement()}")
    println(model)
    val criterion = new CrossEntropyCriterion[Float]()
    val epochNum = 90
    val featureShape = Array(3, 224, 224)
    val targetShape = Array(1)
    val sgd = new SGD[Float]
    val state = T("learningRate" -> 0.1, "momentum" -> 0.9, "dampening" -> 0.0)
    val stageImgs = new util.ArrayDeque[Image](batchSize)
    val input = Tensor[Float](batchSize, 3, 224, 224)
    val target = Tensor[Float](batchSize)
    val meanRFloat = MeanStd.mean(0) // meanR.toFloat
    val meanGFloat = MeanStd.mean(1) // meanG.toFloat
    val meanBFloat = MeanStd.mean(2) // meanB.toFloat
    val varRFloat = MeanStd.std(0) // varR.toFloat
    val varGFloat = MeanStd.std(1) // varG.toFloat
    val varBFloat = MeanStd.std(2) // varB.toFloat

    Split.unSetValFlag

    val iter = ImageNetUtils.toTensorFloat(
      donkey.map(d => {
        stageImgs.push(d._2)
        (labelsMap(d._2.label).toFloat, d._1)
      }),
      featureShape,
      targetShape,
      batchSize,
      (meanRFloat, meanGFloat, meanBFloat),
      (varRFloat, varGFloat, varBFloat),
      input,
      target
    )

    Split.setValFlag

    val stageImgsVal = new util.ArrayDeque[Image](batchSize)
    val iterVal = ImageNetUtils.toTensorFloat(
      donkeyVal.map(d => {
        stageImgsVal.push(d._2)
        (labelsMap(d._2.label).toFloat, d._1)
      }),
      featureShape,
      targetShape,
      batchSize,
      (meanRFloat, meanGFloat, meanBFloat),
      (varRFloat, varGFloat, varBFloat),
      input,
      target
    )

    //log(s"meanR is $meanR meanG is $meanG meanB is $meanB")
    //log(s"varR is $varR varG is $varG varB is $varB")
    log("Start to train...")

    var wallClockTime = 0L
    for (i <- 1 to epochNum) {
      println(s"Epoch[$i] Train")

      /*for (regime <- regimes(netType)) {
        if (i >= regime._1 && i <= regime._2) {
          state("learningRate") = regime._3
          state("weightDecay") = regime._4
        }
      }
      */

      state("learningRate") =  state("learningRate").asInstanceOf[Double] * Math.pow(0.1, math.floor((i-1)/30))
      var j = 0
      var c = 0
      model.training()
      while (j < dataSet.getTotal) {
        val start = System.nanoTime()
        val (input, target) = iter.next()
        val readImgTime = System.nanoTime()
        model.zeroGradParameters()
        val output = model.forward(input)
        val loss = criterion.forward(output, target)
        val gradOutput = criterion.backward(output, target)
        model.backward(input, gradOutput)
        sgd.optimize(_ => (loss, grad), weights, state, state)
        val end = System.nanoTime()
        wallClockTime += end - start
        log(s"Epoch[$i][Iteration $c $j/${dataSet.getTotal}][Wall Clock ${wallClockTime / 1e9}s]" +
          s" loss is $loss time ${(end - start) / 1e9}s read " +
          s"time ${(readImgTime - start) / 1e9}s train time ${(end - readImgTime) / 1e9}s." +
          s" Throughput is ${input.size(1).toDouble / (end - start) * 1e9} img / second")
        while (!stageImgs.isEmpty) {
          dataSet.post(stageImgs.poll())
        }
        j += input.size(1)
        c += 1
      }

      if (i % testInterval == 0) {
        model.evaluate()
        var top1Correct = 0
        var top5Correct = 0
        var k = 0
        while (k < dataSetVal.getTotal) {
          val (input, target) = iterVal.next()
          val output = model.forward(input)
          top1Correct += EvaluateMethods.calcAccuracy(output, target)._1
          top5Correct += EvaluateMethods.calcTop5Accuracy(output, target)._1
          while (!stageImgsVal.isEmpty) {
            dataSetVal.post(stageImgsVal.poll())
          }
          k += input.size(1)
        }

        val top1Accuracy = top1Correct.toDouble / dataSetVal.getTotal
        val top5Accuracy = top5Correct.toDouble / dataSetVal.getTotal
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Top-1 Accuracy is $top1Accuracy")
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Top-5 Accuracy is $top5Accuracy")
        println(s"Save model and state to $modelPath-$i")
        File.save(model, modelPath + s"-$i.model")
        File.save(state, modelPath + s"-$i.state")
      }

      log("shuffle")
      dataSet.shuffle
      log("shuffle end")
    }
  }

  def main(args: Array[String]): Unit = {
    // See http://stackoverflow.com/questions/26535842/multithreaded-jpeg-image-processing-in-java
    Class.forName("javax.imageio.ImageIO")
    Class.forName("java.awt.color.ICC_ColorSpace")
    Class.forName("sun.java2d.cmm.lcms.LCMS")
    ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

    require(args.length == 10, "invalid args, should be <path> <parallelism> <labelPath>" +
      " <testInterval> <netType> <classNum> <dataType> <batchSize> <modelDepth>")

    val path = args(0)
    val parallelism = args(1).toInt
    val labelsMap = ImageNetUtils.getLabels(args(2))
    val pathVal = args(3)
    val testInterval = args(4).toInt
    val netType = args(5)
    val classNum = args(6).toInt
    val batchSize = args(7).toInt
    val modelPath = args(8)
    val modelDepth = args(9).toInt

    val dataSet = new DataSets(path, classNum, labelsMap)
    val donkey = new Donkey(parallelism, dataSet)
    val dataSetVal = new DataSets(pathVal, classNum, labelsMap)
    val donkeyVal = new Donkey(parallelism, dataSetVal)

    log("shuffle")
    dataSet.shuffle
    log("shuffle end")

    run(donkey, dataSet, netType, classNum, labelsMap, testInterval,
      donkeyVal, dataSetVal, batchSize, modelPath, modelDepth)
  }

  def shareGradInput[@specialized(Float, Double) T: ClassTag](model: Module[T])
                                                             (implicit ev: TensorNumeric[T]): Unit = {
    def sharingKey(m: Module[T]) = m.getClass.getName

    val cache = mutable.Map[Any, Storage[T]]()

    model.mapModules(m => {
      val moduleType = m.getClass.getName
      if (!moduleType.equals("com.intel.analytics.sparkdl.nn.ConcatAddTable")) {
        val key = sharingKey(m)
        if (!cache.contains(key)){
          cache.put(key, Storage(Array(ev.fromType[Int](1))))
        }
        m.gradInput = Tensor[T](cache.get(key).get, 1, Array(0))
      }
    })

    for ((m, i) <- model
      .findModules("com.intel.analytics.sparkdl.nn.ConcatAddTable")
      .zipWithIndex){
      if (!cache.contains(i % 2)) {
        cache.put(i % 2, Storage(Array(ev.fromType[Int](1))))
      }
      m.gradInput = Tensor[T](cache.get(i % 2).get, 1, Array(0))
    }

    cache.put("gradWeightMM", Storage(Array(ev.fromType[Int](1))))
    cache.put("fInput", Storage(Array(ev.fromType[Int](1))))
    cache.put("fGradInput", Storage(Array(ev.fromType[Int](1))))
    for ((m, i) <- model
      .findModules("com.intel.analytics.sparkdl.nn.SpatialConvolution")
      .zipWithIndex){
      val tmpModel = m.asInstanceOf[SpatialConvolution[T]]
      tmpModel.setSharedVar
      tmpModel.setGradWeightMM(Tensor[T](cache.get("gradWeightMM").get))
      tmpModel.fInput = Tensor[T](cache.get("fInput").get)
      tmpModel.fGradInput = Tensor[T](cache.get("fGradInput").get)
    }
  }

  def convInit[@specialized(Float, Double) T: ClassTag](name: String, model: Module[T])
                                                       (implicit ev: TensorNumeric[T]): Unit = {
    for ((m, i) <- model
      .findModules(name)
      .zipWithIndex) {
      val tmpModel = m.asInstanceOf[SpatialConvolution[T]]
      val n = tmpModel.kernelW * tmpModel.kernelH * tmpModel.nOutputPlane
      tmpModel.weight.apply1(_ => ev.fromType[Float](RNG.normal(0, Math.sqrt(2 / n)).toFloat))
      tmpModel.bias.apply1(_ => ev.fromType[Float](0))
    }
  }

  def bnInit[@specialized(Float, Double) T: ClassTag](name: String, model: Module[T])
                                                     (implicit ev: TensorNumeric[T]): Unit = {
    for ((m, i) <- model
      .findModules(name)
      .zipWithIndex) {
      val tmpModel = m.asInstanceOf[SpatialBatchNormalization[T]]
      tmpModel.weight.apply1(_ => ev.fromType[Float](1f))
      tmpModel.bias.apply1(_ => ev.fromType[Float](0f))
    }
  }

  object MeanStd {
    val mean = Array(0.485f, 0.456f, 0.406f)
    val std = Array(0.229f, 0.224f, 0.225f)
  }
  object PCA {
    val eigval = Tensor[Float](Storage(Array( 0.2175f, 0.0188f, 0.0045f )), 1, Array(3))
    val eigvec = Tensor[Float](Storage(Array( -0.5675f,  0.7192f,  0.4009f,
                                       -0.5808f, -0.0045f, -0.8140f,
                                       -0.5836f, -0.6948f,  0.4203f)), 1, Array(3, 3))
    val alphastd = 0.1f
    val alpha = Tensor[Float](3).apply1(_ => RNG.uniform(0, alphastd).toFloat)
    val rgb = eigvec.clone.cmul(alpha.view(1, 3).expand(Array(3, 3)))
      .cmul(eigval.view(1, 3).expand(Array(3, 3)))
      .sum(2).squeeze
  }
  object ColorJitter {
    val brightness = 0.4f
    val contrast = 0.4f
    val saturation = 0.4f
    def blend[@specialized(Float, Double) T: ClassTag](img1: Tensor[T], img2: Tensor[T], alpha: T)
                                                      (implicit ev: TensorNumeric[T]) =
      img1.mul(alpha).add(ev.minus(ev.fromType(1), alpha), img2)
    def grayScale[@specialized(Float, Double) T: ClassTag](dst: Tensor[T], img: Tensor[T])
                                                          (implicit ev: TensorNumeric[T]): Tensor[T] = {
      dst.resizeAs(img)
      dst(1).zero
      dst(1).add(ev.fromType(0.299), img(1)).add(ev.fromType(0.587), img(2)).add(ev.fromType(0.114), img(3))
      dst(2).copy(dst(1))
      dst(3).copy(dst(1))
      dst
    }
  }
  object Split {
    var flag = false
    def setValFlag() = flag = true
    def unSetValFlag() = flag = false
    def getValFlag() = flag
  }
}
