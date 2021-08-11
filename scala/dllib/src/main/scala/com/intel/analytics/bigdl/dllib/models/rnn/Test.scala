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


import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.dataset.{DataSet, FixedLength, PaddingParam, SampleToMiniBatch, _}
import com.intel.analytics.bigdl.dataset.text.{Dictionary, LabeledSentence, LabeledSentenceToSample, TextToLabeledSentence}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Loss, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.util.Random

object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)

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

      if (param.evaluate) {
        val valtokens = SequencePreprocess(
          param.folder + "/test.txt",
          sc = sc,
          param.sentFile,
          param.tokenFile).collect()
        val maxValLength = valtokens.map(x => x.length).max

        val totalVocabLength = vocab.getVocabSize() + 1
        val startIdx = vocab.getIndex(SentenceToken.start)
        val endIdx = vocab.getIndex(SentenceToken.end)
        val padFeature = Tensor[Float]().resize(totalVocabLength)
        padFeature.setValue(endIdx + 1, 1.0f)
        val padLabel = Tensor[Float](T(startIdx.toFloat + 1.0f))
        val featurePadding = PaddingParam(Some(Array(padFeature)),
          FixedLength(Array(maxValLength)))
        val labelPadding = PaddingParam(Some(Array(padLabel)),
          FixedLength(Array(maxValLength)))

        val evaluationSet = DataSet.array(valtokens)
          .transform(TextToLabeledSentence[Float](vocab))
          .transform(LabeledSentenceToSample[Float](totalVocabLength))
          .transform(SampleToMiniBatch[Float](param.batchSize,
            Some(featurePadding), Some(labelPadding))).toLocal()

        val result = model.evaluate(evaluationSet,
          Array(new Loss[Float](
            TimeDistributedCriterion[Float](
              CrossEntropyCriterion[Float](),
              sizeAverage = true)).asInstanceOf[ValidationMethod[Float]]))

        result.foreach(r => println(s"${r._2} is ${r._1}"))
      } else {
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
          SampleToMiniBatch[Float](batchSize).apply(iter)
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
      sc.stop()
    }
  }
}
