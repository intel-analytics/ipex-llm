/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.models.embedding

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{MiniBatch, Transformer}
import com.intel.analytics.bigdl.dataset.text.Tokenizer
import com.intel.analytics.bigdl.models.embedding.Utils.Word2VecConfig
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.rdd.RDD

import scala.collection.{mutable, _}


/**
 *  Entry in vocabulary
 */
private case class WordCount(
  var word: String,
  var count: Int
)

object Word2Vec {
  def apply(params: Word2VecConfig): Word2Vec = new Word2Vec(params)
}

class Word2Vec(val params: Word2VecConfig) {
  private var trainWordsCount = 0L
  private var vocabSize = 0
  private var powerSum = 0
  @transient private var vocab: Array[WordCount] = _
  @transient private var word2Id = mutable.HashMap.empty[String, Int]
  @transient private var powerUnigram: Array[Int] = _
  var wordVectors = LookupTable(vocabSize, params.embeddingSize)

  def getModel: Module[Float] = {
    new Sequential()
      .add(wordVectors)
      .add(SplitTable(1))
      .add(ConcatTable()
        .add(Sequential()
          .add(NarrowTable(2, -1))
          .add(JoinTable(1)))
        .add(SelectTable(1)))
      .add(MM(transA = false, transB = true))
      .add(Sigmoid())
  }

  def buildVocab(words: RDD[String]): Unit = {
    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= params.minCount)
      .map(x => WordCount(x._1, x._2))
      .collect()
      .sortWith((a, b) => a.count > b.count)

    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    var id = 0
    while (id < vocabSize) {
      word2Id += vocab(id).word -> id
      trainWordsCount += vocab(id).count
      id += 1
    }

    println(s"vocabSize = $vocabSize, trainWordsCount = $trainWordsCount")
  }

  /**
   * Create power-unigram distribution to generate negative samples
   *
   */
  def buildNegSampleDistribution(): Unit = {
    powerSum = math.round(
      vocab.foldLeft(0.0)((sum, wordCount) => sum + math.pow(wordCount.count, params.alpha))).toInt

    powerUnigram = new Array(powerSum)

    def wordProb(id: Int) = math.pow(vocab(id).count, params.alpha) / powerSum

    var wordId = 1
    var idx = 1
    var powerProb = wordProb(wordId)
    while (idx < powerSum) {
      powerUnigram(idx) = wordId
      if (idx / powerSum > powerProb) {
        wordId += 1
        powerProb = wordProb(wordId)
      }

      if (wordId > powerSum) {
        wordId -= 1
      }

      idx += 1
    }
  }

  /**
   * Sample negative contexts from power unigram distribution,
   * and conbine them with given word and context together as a training sample
   *
   * @param context given context
   * @return a one-dimension tensor, whose first elmement is the given word,
   *         second element is the given context, and the rest elements are the
   *         sampled negative contexts.
   */
  def sampleContexts(word: Int, context: Int): Tensor[Float] = {
    val contexts = Tensor(params.numNegSamples + 2)
    contexts.setValue(1, word)
    contexts.setValue(1, context)
    var i = 3
    while (i <= params.numNegSamples + 2) {
      // Sample a negative context
      val negContext = powerUnigram(math.ceil(RNG.uniform(1, powerSum)).toInt)
      if (context != negContext) {
        contexts.setValue(i, negContext)
        i += 1
      }
    }
    contexts
  }

  def fit[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    val words = dataset.flatMap(x => x)

    buildVocab(words)

    buildNegSampleDistribution()

//    val sc = dataset.context

    (Tokenizer()
      -> WordsToIds(word2Id, params.maxSentenceLength)
      -> WordIdsToMiniBatch(params.numNegSamples, params.windowSize))
  }

  /**
   * Transform a sequence of words to its corresponding ids, and filter the length
   * of each set of words less than `maxSentenceLength`
   *
   */
  case class WordsToIds(
    word2Id: mutable.HashMap[String, Int],
    maxSentenceLength: Int)
    extends Transformer[Seq[String], Seq[Int]] {
    val sentence = mutable.ArrayBuilder.make[Int]

    override def apply(prev: Iterator[Seq[String]]): Iterator[Seq[Int]] =
      prev.map(words => {
        var sentenceLength = 0
        while (sentenceLength < maxSentenceLength) {
          val word = word2Id.get(words(sentenceLength))
          word match {
            case Some(w) =>
              sentence += w
              sentenceLength += 1
            case None =>
          }
          sentenceLength += 1
        }
        sentence.result()
      })
  }

  /**
   * Transform sentence represented as word ids to MiniBatch
   * Each sensence is represented as a batch
   *
   */
  case class WordIdsToMiniBatch(numNegSamples: Int, window: Int)
    extends Transformer[Seq[Int], MiniBatch[Float]] {
    override def apply(prev: Iterator[Seq[Int]]): Iterator[MiniBatch[Float]] = {
      val label = Tensor(numNegSamples + 1).zero()
      label.setValue(1, 1)

      prev.map(sentence => {
        val inputTable = T()
        val labelTable = T()
        sentence.zipWithIndex.foreach {
          case (i, word) =>
            val reducedWindow = RNG.uniform(0, window).toInt
            var j = i - reducedWindow
            j = if (j < 0) 0 else j

            while (j <= i + reducedWindow && j < sentence.length) {
              if (j != i) {
                val context = sentence(j)
                val sample = sampleContexts(word, context)
                inputTable.insert(sample)
                labelTable.insert(label)
              }
            }
        }

        val inputTensor =
          JoinTable(1).updateOutput(inputTable)
        val labelTensor =
          JoinTable(1).updateOutput(labelTable)

        MiniBatch(inputTensor, labelTensor)
      })
    }
  }
}
