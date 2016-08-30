package com.intel.webscaleml.nn.example

import java.nio.ByteBuffer
import java.nio.file.{Paths, Files}

import com.intel.webscaleml.nn.nn._
import com.intel.webscaleml.nn.optim._
import com.intel.webscaleml.nn.tensor.{torch, Tensor, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import com.intel.webscaleml.nn.example.Utils._

import scala.util.Random
import MNIST._

object MNISTParallel {

  def main(args : Array[String]) : Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {train(params)}}.getOrElse{sys.exit(1)}
  }

  def train(params : Params) = {
    val folder = params.folder
    val trainData = loadFile(s"$folder/train-images.idx3-ubyte", s"$folder/train-labels.idx1-ubyte")
    val testData = loadFile(s"$folder/t10k-images.idx3-ubyte", s"$folder/t10k-labels.idx1-ubyte")

    val partitionNum = params.partitionNum
    val conf = new SparkConf().setAppName(s"MNIST with Partition $partitionNum")
    if(params.isLocalSpark) {
      conf.setMaster(s"local[${params.partitionNum}]")
    }
    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_AFFINITY", "compact,1,0")
    conf.setExecutorEnv("MKL_DYNAMIC", "false")
    conf.setExecutorEnv("OMP_NUM_THREADS", params.parallelism.toString)
    conf.set("spark.task.cpus", params.parallelism.toString)
    conf.set("spark.shuffle.blockTransferService", "nio")

    val sc = new SparkContext(conf)
    val trainRDD = sc.parallelize(trainData).repartition(partitionNum).persist()
    val testRDD = trainRDD.zipPartitions(sc.parallelize(testData).repartition(partitionNum))((train, test) => test).persist()
    trainRDD.setName("trainRDD")
    testRDD.setName("testRDD")
    val (mean, std) = computeMeanStd(trainRDD)
    println(s"mean is $mean std is $std")

    val driverConfig = params.masterConfig.clone()

    val workerConfig = params.workerConfig.clone()

    val communicator = new CompressedCommunicator[Double](trainRDD, trainRDD.partitions.length)
    val dataSets = new ShuffleBatchDataSet[Array[Byte], Double](trainRDD, toTensor(mean, std), 10, 40, params.workerConfig[Int]("batchnum"))
    val testDataSets = new ShuffleBatchDataSet[Array[Byte], Double](testRDD, toTensor(mean, std), 10, 40, params.workerConfig[Int]("batchnum"))
    val optimizer = if(params.distribute == "parallel")
      new WeightAvgEpochOptimizer[Double](getModule(params.net), new ClassNLLCriterion(), getOptimMethod(params.masterOptM), communicator, dataSets, driverConfig)
    else if(params.distribute == "serial")
      new GradAggEpochOptimizer[Double](getModule(params.net), new ClassNLLCriterion(), getOptimMethod(params.masterOptM), communicator, dataSets, driverConfig)
    else
       ???

    optimizer.setTestInterval(params.valIter)
    optimizer.setPath("./mnist.model")
    optimizer.setEvaluation(EvaluateMethods.calcAccuracy)
    optimizer.setTestDataSet(testDataSets)

    optimizer.optimize()
  }

  def computeMeanStd(data : RDD[Array[Byte]]) : (Double, Double) = {
    val count = data.count()
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
