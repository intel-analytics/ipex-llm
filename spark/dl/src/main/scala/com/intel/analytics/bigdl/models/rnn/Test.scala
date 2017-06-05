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
import com.intel.analytics.bigdl.dataset.text.{Dictionary, LabeledSentence, LabeledSentenceToSample}
import com.intel.analytics.bigdl.nn.{Concat, Identity, LogSoftMax, Module}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

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

      val vocab = Dictionary(param.folder)
      val conf = Engine.createSparkConf()
        .setAppName("Test rnn on text")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val model = Module.load[Float](param.modelSnapshot.get)

      val timeDim = 2
      val featDim = 3
      val concat = Tensor[Float]()
      val lines = readSentence(param.folder)
      val input = lines.map(x =>
        x.map(t => vocab.getIndex(t).toFloat))
      val labeledInput = input.map(x =>
        new LabeledSentence[Float](x, x))

      val vocabSize = vocab.getVocabSize() + 1
      val batchSize = param.batchSize

      val rdd = sc.parallelize(labeledInput).mapPartitions(iter =>
        LabeledSentenceToSample[Float](vocabSize).apply(iter)
      ).mapPartitions(iter =>
        SampleToBatch[Float](batchSize).apply(iter)
      )

      val flow = rdd.mapPartitions(iter => {
        iter.map(batch => {
          var curInput = batch.getInput().toTensor[Float]
          // Iteratively output predicted words
          for (i <- 1 to param.numOfWords.getOrElse(0)) {
            val input = curInput.max(featDim)._2
            val output = model.forward(curInput).toTensor[Float]
            val predict = output.max(featDim)._2.select(timeDim, output.size(timeDim))
            concat.resize(curInput.size(1), curInput.size(timeDim) + 1, curInput.size(featDim))
            concat.narrow(timeDim, 1, curInput.size(timeDim)).copy(curInput)
            for (j <- 1 to curInput.size(1)) {
              concat.setValue(j, concat.size(timeDim), predict.valueAt(j, 1).toInt + 1, 1.0f)
            }
            curInput = concat
          }
          val predIdx = curInput.max(featDim)._2
          val predArray = new Array[Float](predIdx.nElement())
          Array.copy(predIdx.storage().array(), predIdx.storageOffset() - 1,
            predArray, 0, predIdx.nElement())
          predArray.grouped(predIdx.size(timeDim)).toArray[Array[Float]]
        })
      }).collect().flatMap(x => x)

      val results = flow.map(x => x.map(t => vocab.getWord(t)))
      results.foreach(x => logger.info(x.mkString(" ")))
    }
  }
}
