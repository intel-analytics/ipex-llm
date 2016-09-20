package com.intel.analytics.sparkdl.example

import com.intel.analytics.sparkdl.example.ImageNetUtils._
import com.intel.analytics.sparkdl.example.Utils._
import com.intel.analytics.sparkdl.nn.ClassNLLCriterion
import com.intel.analytics.sparkdl.optim.EpochOptimizer.Regime
import com.intel.analytics.sparkdl.optim.{EvaluateMethods, GradAggEpochOptimizer, ShuffleBatchDataSet, Metrics}
import com.intel.analytics.sparkdl.ps.{AllReduceParameterManager, OneReduceParameterManager}
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.T
import org.apache.hadoop.io.Text
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

object DummyImageNetParallel {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("com.intel.analytics.sparkdl.optim").setLevel(Level.INFO)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {
      train(params)
    }
    }.getOrElse {
      sys.exit(1)
    }
  }

  def train(params: Params) = {
    val conf = new SparkConf().setAppName(s"Test")
    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_BLOCKTIME", "0")
    conf.setExecutorEnv("OMP_WAIT_POLICY", "passive")
    conf.setExecutorEnv("OMP_NUM_THREADS", s"${params.parallelism}")
    conf.set("spark.task.maxFailures", "1")
    conf.set("spark.shuffle.blockTransferService", "nio")
    conf.set("spark.akka.frameSize", "10")  // akka networking speed is slow
    conf.set("spark.task.cpus", params.parallelism.toString)
    val sc = new SparkContext(conf)

    val classNum = 1000
    println("cache data")
    val trainData = sc.parallelize(1 to params.partitionNum * 10000, params.partitionNum).cache()
    trainData.count()
    println("done")
    val criterion = new ClassNLLCriterion[Float]()
    val model = AlexNet.getModel[Float](classNum)
    println(model)
    val parameters = model.getParameters()._1
    val metrics = new Metrics

    val optM = getOptimMethodFloat(params.masterOptM)
    val dataSets = new ShuffleBatchDataSet[Int, Float](
      trainData, (d, t1, t2) => (t1.resize(Array(params.workerConfig[Int]("batch"), 3, 224, 224)),
        t2.resize(Array(params.workerConfig[Int]("batch"))).fill(1)),
      params.workerConfig[Int]("batch"), params.workerConfig[Int]("batch"))
    //val pm = new OneReduceParameterManager[Float](parameters, dataSets.partitions(), metrics)
    val pm = new AllReduceParameterManager[Float](parameters, dataSets.partitions(), metrics)

    val optimizer = new GradAggEpochOptimizer[Float](model, criterion, optM,
      pm, dataSets, metrics, params.masterConfig)
    optimizer.setMaxEpoch(100)

    optimizer.optimize()
  }
}
