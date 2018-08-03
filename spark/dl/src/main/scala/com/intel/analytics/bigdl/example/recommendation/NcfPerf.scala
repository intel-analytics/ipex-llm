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

import com.intel.analytics.bigdl.nn.BCECriterion
import com.intel.analytics.bigdl.optim.{Adam, ParallelAdam}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine

import scala.util.Random

object NcfPerf {
  def main(args: Array[String]): Unit = {
    val iteration = args(0).toInt
    val batchSize = args(1).toInt
    val core = args(2).toInt
    val train = args(3).toInt == 1
    System.setProperty("bigdl.localMode", "true")
    Engine.init(1, core, false)
    val userCount = 138493
    val itemCount = 26744

    val model = NeuralCFV2[Float](userCount, itemCount, 1, 128, 128,
      hiddenLayers = Array(128, 64),
      mfEmbed = 64)
        .buildModel()
    val criterion = BCECriterion[Float]()
    val optimMethod = new Adam[Float]()

    val input = Tensor[Float](batchSize * core, 2)
    input.select(2, 1).apply1(_ => Random.nextInt(userCount) + 1)
    input.select(2, 2).apply1(_ => Random.nextInt(itemCount) + 1)
    val target = Tensor[Float](batchSize * core, 1).apply1(_ => Random.nextInt(2))

    val modelForwardTime = new Array[Long](core)
    val modelBackwardTime = new Array[Long](core)
    val criterionForwardTime = new Array[Long](core)
    val criterionBackwardTime = new Array[Long](core)
    var accgradientTime = 0L
    var updateWeightTime = 0L


    val (w, g) = model.getParameters()
    println(s"model weight length ${w.nElement()}")
    val workingModels = (0 until core).map{i =>
      val newmodel = model.cloneModule()
      newmodel.getParameters()._1.set(w)
      newmodel
    }
    val workingModelWAndG = workingModels.map(_.getParameters())

    val subModelNumber = core
    val parallelism = core
    val gradLength = g.nElement()
    val syncGradTaskSize = gradLength / subModelNumber
    val syncGradExtraTask = gradLength % subModelNumber
    val syncGradParallelNum =
      if (syncGradTaskSize == 0) syncGradExtraTask else subModelNumber

    // warm up
    (0 until 5).foreach{i =>
      Engine.default.invokeAndWait((0 until core).map { tid =>
        () =>
          val currentInput = input.narrow(1, tid * batchSize + 1, batchSize)
          val currentTarget = target.narrow(1, tid * batchSize + 1, batchSize)
          val currentModel = workingModels(tid)

          val output = currentModel.forward(currentInput)
          val loss = criterion.forward(output, currentTarget)
          val gradOutput = criterion.backward(output, currentTarget)
          val gradInput = currentModel.backward(currentInput, gradOutput)

      })
    }

    if (train) {
      (0 until iteration).foreach { i =>
        input.select(2, 1).apply1(_ => Random.nextInt(userCount) + 1)
        input.select(2, 2).apply1(_ => Random.nextInt(itemCount) + 1)
        target.apply1(_ => Random.nextInt(2))
        println(i)

        Engine.default.invokeAndWait((0 until core).map { tid =>
          () =>
            val currentInput = input.narrow(1, tid * batchSize + 1, batchSize)
            val currentTarget = target.narrow(1, tid * batchSize + 1, batchSize)
            val currentModel = workingModels(tid)


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

        var start = System.nanoTime()
        val grad = g
        Engine.default.invokeAndWait(
          (0 until syncGradParallelNum).map(tid =>
            () => {
              val offset = tid * syncGradTaskSize + math.min(tid, syncGradExtraTask)
              val length = syncGradTaskSize + (if (tid < syncGradExtraTask) 1 else 0)
              var i = 0
              while (i < parallelism) {
                val sliceG = workingModelWAndG(i)._2.narrow(1, offset + 1, length)
                if (i == 0) {
                  grad.narrow(1, offset + 1, length)
                    .copy(sliceG)
                  sliceG.zero()
                } else {
                  grad.narrow(1, offset + 1, length)
                    .add(sliceG)
                  sliceG.zero()
                }
                i += 1
              }
            })
        )
        grad.div(parallelism)
        accgradientTime += System.nanoTime() - start

        start = System.nanoTime()
        optimMethod.optimize(_ => (1, grad), w)
        updateWeightTime += System.nanoTime() - start
      }

      println(s"${modelForwardTime.max / 1e6 / iteration}ms")
      println(s"${criterionForwardTime.max / 1e6 / iteration}ms")
      println(s"${criterionBackwardTime.max / 1e6 / iteration}ms")
      println(s"${modelBackwardTime.max / 1e6 / iteration}ms")
      println(s"${accgradientTime / 1e6 / iteration}ms")
      println(s"${updateWeightTime / 1e6 / iteration}ms")
    } else {
      var computingTime = 0L
      (0 until iteration).foreach { i =>
        input.select(2, 1).apply1(_ => Random.nextInt(userCount) + 1)
        input.select(2, 2).apply1(_ => Random.nextInt(itemCount) + 1)
        target.apply1(_ => Random.nextInt(2))
        println(i)

        var start = System.nanoTime()
        Engine.default.invokeAndWait((0 until core).map { tid =>
          () =>
            val currentInput = input.narrow(1, tid * batchSize + 1, batchSize)
            val currentTarget = target.narrow(1, tid * batchSize + 1, batchSize)
            val currentModel = workingModels(tid)

            val output = currentModel.forward(currentInput)
            modelForwardTime(tid) += System.nanoTime() - start
        })
        computingTime += System.nanoTime() - start
      }

      println(s"Throughput is ${batchSize * core * iteration * 1e9 / computingTime} records/s")
    }

  }

}
