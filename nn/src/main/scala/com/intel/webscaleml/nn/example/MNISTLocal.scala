package com.intel.webscaleml.nn.example

import java.nio.ByteBuffer
import java.nio.file.{Paths, Files}

import com.intel.webscaleml.nn.nn._
import com.intel.webscaleml.nn.optim._
import com.intel.webscaleml.nn.tensor.{T, Table, Tensor, torch}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser
import com.intel.webscaleml.nn.example.MNIST._
import com.intel.webscaleml.nn.example.Utils._

import scala.util.Random

object MNISTLocal {

  def main(args : Array[String]) : Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {train(params)}}.getOrElse{sys.exit(1)}
    println("main done")
  }

  def shuffle[T](data : Array[T]) = {
    var i = 0
    while(i < data.length) {
      val exchange = i + Random.nextInt(data.length - i)
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
  }

  def train(params : Utils.Params) = {
    val folder = params.folder
    val trainData = loadFile(s"$folder/train-images.idx3-ubyte", s"$folder/train-labels.idx1-ubyte")
    val testData = loadFile(s"$folder/t10k-images.idx3-ubyte", s"$folder/t10k-labels.idx1-ubyte")

    val module = getModule(params.net)
    val optm = getOptimMethod(params.masterOptM)
    val critrion = new ClassNLLCriterion[Double]()
    val (w, g) = module.getParameters()
    var e = 0
    val config = params.masterConfig.clone()
    config("dampening") = 0.0
    var wallClockTime = 0L
    val (mean, std) = computeMeanStd(trainData)
    println(s"mean is $mean std is $std")
    val input = torch.Tensor[Double]()
    val target = torch.Tensor[Double]()
    while(e < config.get[Int]("epoch").get) {
      shuffle(trainData)
      var trainLoss = 0.0
      var i = 0
      var c = 0
      while(i < trainData.length) {
        val start = System.nanoTime()
        val batch = math.min(params.workerConfig[Int]("batch"), trainData.length - i)
        val buffer = new Array[Array[Byte]](batch)
        var j = 0
        while(j < buffer.length) {
          buffer(j) = trainData(i + j)
          j += 1
        }

        toTensor(mean, std)(buffer, input, target)
        module.zeroGradParameters()
        val output = module.forward(input)
        val loss = critrion.forward(output, target)
        val gradOutput = critrion.backward(output, target)
        module.backward(input, gradOutput)
        optm.optimize(_ => (loss, g), w, config, config)
        val end = System.nanoTime()
        trainLoss += loss
        wallClockTime += end - start
        println(s"[Wall Clock ${wallClockTime.toDouble / 1e9}s][epoch $e][iteration $c $i/${trainData.length}] Train time is ${(end - start) / 1e9}seconds. loss is $loss Throughput is ${buffer.length.toDouble / (end - start) * 1e9} records / second")

        i += buffer.length
        c += 1
      }
      println(s"[epoch $e][Wall Clock ${wallClockTime.toDouble / 1e9}s] Training Loss is ${trainLoss / c}")

      // eval
      var k = 0
      var correct = 0
      var count = 0
      var testLoss = 0.0
      val buffer1 = torch.Tensor[Double]()
      val buffer2 = torch.Tensor[Double]()
      while(k < testData.length) {
        val (input, target) = toTensor(mean, std)(Array(testData(k)), buffer1, buffer2)
        val output = module.forward(input)
        testLoss += critrion.forward(output, target)
        val (curCorrect, curCount) = EvaluateMethods.calcAccuracy(output, target)
        correct += curCorrect
        count += curCount
        k += 1
      }
      println(s"[Wall Clock ${wallClockTime.toDouble / 1e9}s] Test Loss is ${testLoss / k} Accuracy is ${correct.toDouble / count}")

      e += 1
    }
    println("train done")
  }

  def computeMeanStd(data : Array[Array[Byte]]) : (Double, Double) = {
    val count = data.length
    val total = data.map(img => {
      var i = 1
      var sum = 0.0
      while(i < img.length) {
        sum += (img(i) & 0xff) / 255.0
        i += 1
      }

      sum
    }).reduce(_ + _)

    val mean = total / (count * rowN * colN)

    val stdTotal = data.map(img => {
      var i = 1
      var stdSum = 0.0
      while(i < img.length) {
        val s = (img(i) & 0xff) / 255.0 - mean
        stdSum += s * s
        i += 1
      }

      stdSum
    }).reduce(_ + _)

    val std = math.sqrt(stdTotal / (count * rowN * colN))

    (mean, std)
  }
}
