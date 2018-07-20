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

package com.intel.analytics.bigdl.example.recommendation

import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine

import scala.util.Random

object AdamPerf {
  def main(args: Array[String]): Unit = {
    val iteration = args(0).toInt
    val batchSize = args(1).toInt
    val core = args(2).toInt
    System.setProperty("bigdl.localMode", "true")
    Engine.init(1, core, false)
    val userCount = 138493
    val itemCount = 26744

    val model = NeuralCFV2[Float](userCount, itemCount, 1, 128, 128,
      hiddenLayers = Array(128, 64),
      mfEmbed = 64)
      .buildModel()
    val (w, g) = model.getParameters()
    val optimMethod = new Adam[Float]()
    g.randn()

    // warm up
    (0 until 5).foreach{i =>
      optimMethod.optimize(_ => (1, g), w)
    }

    var count = 0L
    (0 until iteration).foreach { i =>
      println(i)
      g.randn()
      val start = System.nanoTime()
      optimMethod.optimize(_ => (1, g), w)
      val end = System.nanoTime()
      println(s"time is ${(end - start) / 1e6.toLong}")
      count += end - start
    }
  }

}
