package com.intel.analytics.bigdl.ppml.example.benchmark

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.FLContext
import com.intel.analytics.bigdl.ppml.algorithms.vfl.FGBoostRegression
import com.intel.analytics.bigdl.ppml.utils.TimingSupportive
import scopt.OptionParser

import scala.util.Random


object FGBoostBenchmark extends TimingSupportive {
  case class Params(dataSize: Int = 100, dataDim: Int = 10, numRound: Int = 10)
  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[Params]("Text Classification Example") {
      opt[Int]("dataSize")
        .text("data size")
        .action((x, params) => params.copy(dataSize = x))
      opt[Int]("dataDim")
        .text("data dimension")
        .action((x, params) => params.copy(dataDim = x))
      opt[Int]("numRound")
        .text("boosting round numer")
        .action((x, params) => params.copy(numRound = x))
    }

    val param = parser.parse(args, Params()).get

    FLContext.initFLContext()
    val random = new Random()
    val data = (0 until param.dataSize).map(_ => Tensor[Float](param.dataDim).rand()).toArray
    val label = (0 until param.dataSize).map(_ => random.nextFloat()).toArray
    val fGBoostRegression = new FGBoostRegression()
    timing("train") {
      fGBoostRegression.fit(data, label, param.numRound)
    }
    timing("predict") {
      val fGBoostResult = fGBoostRegression.predict(data).map(tensor => tensor.value())
    }

  }
}
