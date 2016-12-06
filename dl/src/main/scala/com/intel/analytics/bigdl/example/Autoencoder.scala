package com.intel.analytics.sparkdl.example

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.sparkdl.example.MNIST._
import com.intel.analytics.sparkdl.example.Utils._
import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.optim._
import com.intel.analytics.sparkdl.tensor.Tensor
import org.apache.log4j.{Level, Logger}

import scala.util.Random

object Autoencoder {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {
      train(params)
    }
    }.getOrElse {
      sys.exit(1)
    }
    println("main done")
  }

  private def shuffle[T](data: Array[T]) = {
    var i = 0
    while (i < data.length) {
      val exchange = i + Random.nextInt(data.length - i)
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
  }

  def getAE(netType: String)(): Module[Tensor[Double], Tensor[Double], Double] = {
    netType.toLowerCase match {
      case "ae" =>
        val model = new Sequential[Tensor[Double], Tensor[Double], Double]
        model.add(new Reshape(Array(featureSize)))
        model.add(new Linear(featureSize, 32))
        model.add(new ReLU[Double]())
        model.add(new Linear(32, featureSize))
        model.add(new Sigmoid[Double]())
        model
      case _ =>
        throw new UnsupportedOperationException
    }
  }

  def computeMeanStd(data: Array[Array[Byte]]): (Double, Double) = {
    val count = data.length
    val total = data.map(img => {
      var i = 1
      var sum = 0.0
      while (i < img.length) {
        sum += (img(i) & 0xff) / 255.0
        i += 1
      }

      sum
    }).reduce(_ + _)

    val mean = total / (count * rowN * colN)

    val stdTotal = data.map(img => {
      var i = 1
      var stdSum = 0.0
      while (i < img.length) {
        val s = (img(i) & 0xff) / 255.0 - mean
        stdSum += s * s
        i += 1
      }

      stdSum
    }).reduce(_ + _)

    val std = math.sqrt(stdTotal / (count * rowN * colN))

    (mean, std)
  }

  def toAETensor(mean: Double, std: Double)(inputs: Seq[Array[Byte]], input: Tensor[Double],
                                          target: Tensor[Double]): (Tensor[Double], Tensor[Double]) = {
    val size = inputs.size
    input.resize(Array(size, featureSize))
    target.resize(Array(size, featureSize))
    var i = 0
    while (i < size) {
      val img = inputs(i)
      var j = 0
      while (j < featureSize) {
        input.setValue(i + 1, j % featureSize + 1, (img(j + 1) & 0xff) / 255.0)
        j += 1

        target.setValue(i + 1, j % featureSize + 1, (img(j) & 0xff) / 255.0)
      }
      i += 1
    }

    (input, target)
  }

  private def train(params: Utils.Params) = {
    val folder = params.folder
    val trainData = loadFile(s"$folder/train-images-idx3-ubyte", s"$folder/train-labels-idx1-ubyte")
    val testData = loadFile(s"$folder/t10k-images-idx3-ubyte", s"$folder/t10k-labels-idx1-ubyte")

    val module = getAE("AE")
    val optm = getOptimMethod(params.masterOptM)
    val critrion = new MSECriterion[Double]()
    val (w, g) = module.getParameters()
    var e = 0
    val config = params.masterConfig.clone()
    config("dampening") = 0.0
    var wallClockTime = 0L
    val (mean, std) = computeMeanStd(trainData)
    println(s"mean is $mean std is $std")
    val input = Tensor[Double]()
    val target = Tensor[Double]()
    while (e < 20) {
      shuffle(trainData)
      var trainLoss = 0.0
      var i = 0
      var c = 0
      while (i < trainData.length) {
        val start = System.nanoTime()
        val batch = math.min(150, trainData.length - i)
        val buffer = new Array[Array[Byte]](batch)
        var j = 0
        while (j < buffer.length) {
          buffer(j) = trainData(i + j)
          j += 1
        }

        toAETensor(mean, std)(buffer, input, target)
        module.zeroGradParameters()
        val output = module.forward(input)
        val loss = critrion.forward(output, target)
        val gradOutput = critrion.backward(output, target)
        module.backward(input, gradOutput)
        optm.optimize(_ => (loss, g), w, config, config)
        val end = System.nanoTime()
        trainLoss += loss
        wallClockTime += end - start
        println(s"[Wall Clock ${wallClockTime.toDouble / 1e9}s][epoch $e][iteration" +
          s" $c $i/${trainData.length}] Train time is ${(end - start) / 1e9}seconds." +
          s" loss is $loss Throughput is ${buffer.length.toDouble / (end - start) * 1e9} " +
          s"records / second")

        i += buffer.length
        c += 1
      }
      println(s"[epoch $e][Wall Clock ${wallClockTime.toDouble / 1e9}s]" +
        s" Training Loss is ${trainLoss / c}")

      e += 1
    }
  }
}