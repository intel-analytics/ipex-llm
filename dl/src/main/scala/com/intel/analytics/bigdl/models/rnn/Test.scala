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

package com.intel.analytics.bigdl.models.rnn


import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet, MiniBatch, SampleToBatch}
import com.intel.analytics.bigdl.dataset.text.{LabeledSentence, LabeledSentenceToSample}
import com.intel.analytics.bigdl.nn.{LogSoftMax, Module}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}

import scala.util.Random

object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Utils._
  val logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    testParser.parse(args, new TestParams()).foreach { param =>

      Engine.init
      Engine.setCoreNumber(1)
      val vocab = new Dictionary(param.folder)

      val model = Module.load[Float](param.modelSnapshot.get)

      val logSoftMax = LogSoftMax[Float]()
      val lines = readSentence(param.folder)
      val input = lines.map(x =>
      x.map(t => vocab.getIndex(t).toFloat))

      var labeledInput = input.map(x =>
        new LabeledSentence[Float](x, x))

      val batchSize = 1

      var index = 0
      while (index < param.numOfWords.getOrElse(0)) {
        index += 1

        val validationSet = DataSet.array(labeledInput)
          .transform(LabeledSentenceToSample(vocab.length + 1))
          .transform(SampleToBatch(batchSize = batchSize))
          .asInstanceOf[LocalDataSet[MiniBatch[Float]]]

        val dataIter = validationSet.data(train = false)
        val predict = dataIter.map(batch => {
          require(batch.data.size(1) == 1, "predict sentence one by one")
          val output = model.forward(batch.data)
            .asInstanceOf[Tensor[Float]]
          val predictProbDist = logSoftMax.forward(output(output.size(1)))
            .storage().map(x => math.exp(x).toFloat).toArray
            .map {
              var s = 0.0f; d => {
                s += d; s
              }
            }
            .filter(_ < Random.nextFloat())
          (predictProbDist.length - 1).toFloat
        }).toArray
        labeledInput = (labeledInput zip predict).map(x => {
          val addedInput = x._1.asInstanceOf[LabeledSentence[Float]]
            .data() ++ Array(x._2)
          new LabeledSentence[Float](addedInput, addedInput)
        })
      }

      val results = labeledInput.map(x => x.data()
        .map(t => vocab.getWord(t)))
      results.foreach(x => logger.info(x.mkString(",")))
    }
  }
}
