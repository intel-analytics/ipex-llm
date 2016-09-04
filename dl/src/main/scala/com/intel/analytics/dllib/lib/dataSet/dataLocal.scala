package com.intel.analytics.dllib.lib.dataSet

import java.awt.color.ColorSpace
import java.util
import java.util.Collections
import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.dllib.lib.dataSet.Config
import com.intel.analytics.dllib.lib.dataSet.Utils._
import com.intel.analytics.dllib.lib.example.MNIST._
import com.intel.analytics.dllib.lib.example._
import com.intel.analytics.dllib.lib.nn.ClassNLLCriterion
import com.intel.analytics.dllib.lib.optim.{Adagrad, EvaluateMethods, LBFGS, SGD}
import com.intel.analytics.dllib.lib.tensor.{T, Tensor, torch}
import org.apache.spark.rdd.RDD
import com.intel.analytics.dllib.lib.dataSet.{ImageUtils, toTensor, ComputeMeanStd}

import scala.util.Random


/**
  * Created by zhangli on 16-8-25.
  */
object dataLocal {
  val startTime = System.nanoTime()
  def log(msg : String) : Unit = {
    println(s"[${(System.nanoTime() - startTime) / 1e9}s] $msg")
  }

  def main(args : Array[String]) : Unit = {
    Class.forName("javax.imageio.ImageIO")
    Class.forName("java.awt.color.ICC_ColorSpace")
    Class.forName("sun.java2d.cmm.lcms.LCMS")
    ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

    val parser = getParser()
    parser.parse(args, defaultParams).map { params => {run(params)}}.getOrElse{sys.exit(1)}
    println("datasetLocal done")
  }

  def run(params : Utils.Params)= {
    val trainFiles = params.folder + "/train"
    val testFiles = params.folder + "/val"
    val trainLabel = params.labelsFile + "/trainLabel"
    val testLabel = params.labelsFile + "/testLabel"
    val batchSize = params.workerConfig[Int]("batch")
    val labelShape = Array(1)


    var dataSetConfig = Config.getDataSet(params.dataSetType, params)
    if(!dataSetConfig.supportModels.contains(params.net)){
      sys.exit(1)
    }

    var dataSet = dataSetConfig.loadMethod match{
      case "loadBinaryFile" => ImageUtils.loadBinaryFile(trainFiles, trainLabel)
      case "loadPrimaryImage" => ImageUtils.loadPrimaryImage(trainFiles, trainLabel, 1300)
      case _ => ???
    }
    var dataVal = dataSetConfig.loadMethod match{
      case "loadBinaryFile" => ImageUtils.loadBinaryFile(testFiles, testLabel)
      case "loadPrimaryImage" => ImageUtils.loadPrimaryImage(testFiles, testLabel, 1300)
      case _ => ???
    }

    var (mean,std) = dataSetConfig.meanStd match{
      case "rgbMeanStd" => ComputeMeanStd.rgbMeanStd(dataSet)
      case "grayMeanStd" => ComputeMeanStd.grayMeanStd(dataSet)
      case _ => ???
    }

    var featureShape = dataSetConfig.featureShape
    val input = torch.Tensor[Double](batchSize,featureShape(0),featureShape(1),featureShape(2))
    val target = torch.Tensor[Double](batchSize)
    var iter = dataSetConfig.Tensor match{
      case "toTensorForGray" => toTensor.toTensorForGray(dataSet, featureShape, labelShape, batchSize, mean(0),std(0), input, target)
      case "toTensorRGB" => toTensor.toTensorRGB(dataSet, featureShape, labelShape, batchSize, (mean(0),mean(1),mean(2)), (std(0),std(1),std(2)), input, target)
      case _ => ???
    }

    var iterVal = dataSetConfig.Tensor match{
      case "toTensorForGray" => toTensor.toTensorForGray(dataVal, featureShape, labelShape, batchSize, mean(0),std(0), input, target)
      case "toTensorRGB" => toTensor.toTensorRGB(dataVal, featureShape, labelShape, batchSize, (mean(0),mean(1),mean(2)), (std(0),std(1),std(2)), input, target)
      case _ => ???
    }


    val dataSize = dataSet.length //???
    val valSize = dataVal.length  //???
    train(iter, iterVal, dataSize, valSize, params)

  }

  def train(iter : Iterator[(Tensor[Double], Tensor[Double])], iterVal : Iterator[(Tensor[Double], Tensor[Double])],  dataSize : Int, valSize: Int, params : Utils.Params): Unit ={

    val epochNum = 2 //params.masterConfig[Double]("epoch").toInt
    val testInterval = params.testInterval
    val modelType =  params.net
    val model = Config.getModel[Double](classNum)(modelType)
    val (weights, grad) = model.getParameters()
    val criterion = new ClassNLLCriterion[Double]()

    val optm = params.masterOptM match {
      case "adagrad" => new Adagrad[Double]()
      case "sgd" => new SGD[Double]()
      case "lbfgs" => new LBFGS[Double]()
      case _ =>
        throw new UnsupportedOperationException(s"optim ${params.distribute}")
    }
    //???
    val state = T("momentum" -> params.masterConfig[Int]("momentum"),
      "dampening" -> params.masterConfig[Double]("dampening"))

    var wallClockTime = 0L

    for(i <- 1 to epochNum) {
      println(s"Epoch[$i] Train")
      for(regime <- Config.regimes(modelType)) {
        if(i >= regime._1 && i <= regime._2) {
          state("learningRate") = regime._3
          state("weightDecay") = regime._4
        }
      }
      var j = 0
      var c = 0
      model.training()
      while(j <  dataSize) {
      val start = System.nanoTime()
        val (input, target) = iter.next()
        val readImgTime = System.nanoTime()
        model.zeroGradParameters()
        val output = model.forward(input)
        val loss = criterion.forward(output, target)
        val gradOutput = criterion.backward(output, target)
        model.backward(input, gradOutput)
        optm.optimize(_ => (loss, grad), weights, state, state)
        val end = System.nanoTime()
        wallClockTime += end - start
        log(s"Epoch[$i][Iteration $c $j/${dataSize}][Wall Clock ${wallClockTime / 1e9}s] loss is $loss time ${(end - start) / 1e9}s read " +
          s"time ${(readImgTime - start) / 1e9}s train time ${(end - readImgTime) / 1e9}s. Throughput is ${ input.size(1).toDouble / (end - start) * 1e9} img / second")
        j += input.size(1)
        c += 1
      }

      if(i % testInterval == 0) {
        model.evaluate()
        var correct = 0
        var k = 0
        while(k < valSize) {
          val (input, target) = iterVal.next()
          val output = model.forward(input)
          output.max(2)._2.squeeze().map(target, (a, b) => {
            if(a == b) {
              correct += 1
            }
            a
          })
          k += input.size(1)
        }

        val accuracy = correct.toDouble / valSize
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Accuracy is $accuracy")
      }
    }
  }
}
