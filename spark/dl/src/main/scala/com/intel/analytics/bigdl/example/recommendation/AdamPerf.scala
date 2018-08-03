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

import com.intel.analytics.bigdl.nn.LookupTable
import com.intel.analytics.bigdl.optim.{Adam, EmbeddingAdam}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.util.Random

object AdamPerf {
  def main(args: Array[String]): Unit = {
    val iteration = args(0).toInt
    val batchSize = args(1).toInt
    val core = args(2).toInt
    System.setProperty("bigdl.localMode", "true")
    val sparse = args(3) == "1"
    val rand = args(4) == "1"
    Engine.init(1, core, false)
    val userCount = 138493
    val itemCount = 26744

    val model = NeuralCFV2[Float](userCount, itemCount, 1, 128, 128,
      hiddenLayers = Array(128, 64),
      mfEmbed = 64)
      .buildModel()

    if (sparse) {
      val embeddings = model.embeddingModel.findModules("LookupTable")
        .map(_.asInstanceOf[LookupTable[Float]])
      val input = Tensor[Float].range(1, batchSize * core)
      val embeddingsGradient = embeddings.map{embedding =>
        val inputAndGradient = Array.tabulate(core)(c =>
          (input.narrow(1, batchSize * c + 1, batchSize),
            Tensor[Float](batchSize, embedding.nOutput).rand()))
        val optimMethod = new EmbeddingAdam[Float]()
          optimMethod.setNOutput(embedding.nIndex, embedding.nOutput)
        val parameter = embedding.getParameters()._1
        val parameterArray = Array.tabulate(embedding.nIndex)(i =>
          embedding.weight.select(1, i + 1)
        )
        (inputAndGradient, optimMethod, parameter, parameterArray)
      }

      def update(): Unit = {
        var i = 0
        while (i < embeddingsGradient.size) {
          val v = embeddingsGradient(i)
          val inputAndGradient = v._1
          val optimMethod = v._2
          val parameter = v._3
          val parameterArray = v._4

          var start = System.nanoTime()
//          optimMethod.updateNograd(input, parameterArray)
//          println(s"${i}update parameter array ${parameterArray.length} " +
//            s"Nograd ${System.nanoTime() - start}")
          optimMethod.updateNograd(input, parameter)
          println(s"${i}update parameter ${parameter.nElement()} " +
            s"Nograd ${System.nanoTime() - start}")
          start = System.nanoTime()
          optimMethod.optimizeEmbedding(inputAndGradient, parameter)
          println(s"${i}update parameter ${parameter.nElement()} " +
            s"Embedding ${System.nanoTime() - start}")

          i += 1
        }

      }

      // warm up
      (0 until 20).foreach { i =>
        val n = i % 10
        if (rand) input.range(1 + n * batchSize * core, (n + 1) * batchSize * core)
        update()
      }

      var count = 0L
      (0 until iteration).foreach { i =>
        val n = i % 10
        if (rand) input.range(1 + n * batchSize * core, (n + 1) * batchSize * core)
        println(i)
        val start = System.nanoTime()
        update()
        val end = System.nanoTime()
        println(s"sparse time is ${(end - start) / 1e6.toLong}")
        count += end - start
      }

      println(s"average sparse time is ${count / 1e6.toLong / iteration}")

    } else {
      // update with dense gradient
      val (w, g) = model.embeddingModel.getParameters()
      val optimMethod = new Adam[Float]()
      g.randn()

      // warm up
      (0 until 5).foreach { i =>
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

}
