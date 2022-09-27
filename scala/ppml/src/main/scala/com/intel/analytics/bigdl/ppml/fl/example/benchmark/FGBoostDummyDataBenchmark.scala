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

package com.intel.analytics.bigdl.ppml.fl.example.benchmark

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.fl.algorithms.FGBoostRegression
import com.intel.analytics.bigdl.ppml.fl.FLContext
import com.intel.analytics.bigdl.ppml.fl.utils.TimingSupportive
import scopt.OptionParser

import scala.util.Random


object FGBoostDummyDataBenchmark extends TimingSupportive {
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

    FLContext.initFLContext(1)
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
