/*
 * Copyright 2018 Analytics Zoo Authors.
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

import com.intel.analytics.bigdl.nn.BCECriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine

import scala.util.Random

object NcfPerf {
  def main(args: Array[String]): Unit = {
    val iteration = args(0).toInt
    val batchSize = args(1).toInt
    val core = args(2).toInt
    System.setProperty("bigdl.localMode", "true")
    Engine.init(1, core, false)
    val userCount = 138493
    val itemCount = 26744

    val model = NeuralCFV2[Float](userCount, itemCount, 1, 64, 128, hiddenLayers = Array(128, 64))
        .buildModel()
    val criterion = BCECriterion[Float]()

    val input = Tensor[Float](batchSize * core, 2)
    input.select(2, 1).apply1(_ => Random.nextInt(userCount) + 1)
    input.select(2, 2).apply1(_ => Random.nextInt(itemCount) + 1)
    val target = Tensor[Float](batchSize * core, 1).apply1(_ => Random.nextInt(2))

    var modelForwardTime = new Array[Long](core)
    var modelBackwardTime = new Array[Long](core)
    var criterionForwardTime = new Array[Long](core)
    var criterionBackwardTime = new Array[Long](core)
    val (w, g) = model.getParameters()
    val models = (0 until core).map{i =>
      val newmodel = model.cloneModule()
      newmodel.getParameters()._1.set(w)
      newmodel
    }

    (0 until iteration).foreach { i =>
      input.select(2, 1).apply1(_ => Random.nextInt(userCount) + 1)
      input.select(2, 2).apply1(_ => Random.nextInt(itemCount) + 1)
      target.apply1(_ => Random.nextInt(2))

      Engine.default.invokeAndWait((0 until core).map { tid =>
        () =>
          val currentInput = input.narrow(1, tid * batchSize + 1, batchSize)
          val currentTarget = target.narrow(1, tid * batchSize + 1, batchSize)
          val currentModel = models(tid)


          var start = System.nanoTime()

          val output = currentModel.forward(currentInput)
          modelForwardTime(tid) += System.nanoTime() - start

          start = System.nanoTime()
          val loss = criterion.forward(output, currentTarget)
          criterionForwardTime(tid) += System.nanoTime() - start

          start = System.nanoTime()
          val gradOutput = criterion.backward(output, currentTarget)
          criterionBackwardTime(tid) += System.nanoTime() - start

          start = System.nanoTime()
          val gradInput = currentModel.backward(currentInput, gradOutput)
          modelBackwardTime(tid) += System.nanoTime() - start

      })
    }

    println(s"${modelForwardTime.sum / 1e6 / iteration}ms")
    println(s"${criterionForwardTime.sum / 1e6 / iteration}ms")
    println(s"${criterionBackwardTime.sum / 1e6 / iteration}ms")
    println(s"${modelBackwardTime.sum / 1e6 / iteration}ms")




  }

}
