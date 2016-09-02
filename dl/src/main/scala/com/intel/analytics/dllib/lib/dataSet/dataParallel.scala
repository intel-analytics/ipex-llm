package com.intel.analytics.dllib.lib.dataSet

import com.intel.analytics.dllib.lib.dataSet.{Config, Utils}
import com.intel.analytics.dllib.lib.dataSet.Utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf

/**
  * Created by lzhang2 on 9/2/2016.
  */
object dataParallel {
  def main(args : Array[String]) : Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("com.intel.analytics.dllib.lib.optim").setLevel(Level.INFO)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {run(params)}}.getOrElse{sys.exit(1)}
  }

  def run(params : Utils.Params): Unit = {
    val trainFiles = params.folder + "/train"
    val testFiles = params.folder + "/validation"
    val trainLabel = params.labelsFile + "/trainLabel"
    val testLabel = params.labelsFile + "/testLabel"

    val partitionNum = params.partitionNum
    val optType = params.masterOptM
    val netType = params.net
    val classNum = params.classNum

    val conf = new SparkConf().setAppName(s"ImageNet class: ${params.classNum} Parallelism: ${params.parallelism.toString}, partition : ${params.partitionNum}, " +
      s"masterConfig: ${params.masterConfig}, workerConfig: ${params.workerConfig}")

    //ImageNet
    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_BLOCKTIME", "0")
    conf.setExecutorEnv("OMP_WAIT_POLICY", "passive")
    conf.setExecutorEnv("OMP_NUM_THREADS", s"${params.parallelism}")
    conf.set("spark.task.maxFailures", "1")
    conf.set("spark.shuffle.blockTransferService", "nio")
    conf.set("spark.akka.frameSize", "10")  // akka networking speed is slow
    //MNIST
    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_AFFINITY", "compact,1,0")
    conf.setExecutorEnv("MKL_DYNAMIC", "false")
    conf.setExecutorEnv("OMP_NUM_THREADS", params.parallelism.toString)
    conf.set("spark.task.cpus", params.parallelism.toString)
    conf.set("spark.shuffle.blockTransferService", "nio")
    //Cifar
    conf.setExecutorEnv("OPENBLAS_MAIN_FREE", "1")
    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_AFFINITY", "compact,1,0")
    conf.setExecutorEnv("MKL_DYNAMIC", "false")
    conf.setExecutorEnv("OMP_NUM_THREADS", s"${params.parallelism}")

  }
}
