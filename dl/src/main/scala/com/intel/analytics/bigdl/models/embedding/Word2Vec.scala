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

import breeze.linalg.argsort
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, SampleToBatch, Transformer}
import com.intel.analytics.bigdl.models.embedding.Utils.Word2VecConfig
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.{mutable, _}


/**
 *  Entry in vocabulary
 */
case class WordCount(
  var word: String,
  var count: Int
)

object Word2Vec {
  def apply(params: Word2VecConfig): Word2Vec = new Word2Vec(params)
}

class Word2Vec(val params: Word2VecConfig)
  extends Serializable {
  private var trainWordsCount = 0L
  private var vocabSize = 0
  private var powerSum = 0
  private val minCount = params.minCount
  @transient private var vocab: Array[WordCount] = _
  @transient private var word2Id = mutable.HashMap.empty[String, Int]
  @transient private var powerUnigram: Array[Int] = _
  var wordVectors: LookupTable[Float] = _
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  def getModel: Module[Float] = {
    wordVectors = LookupTable(vocabSize, params.embeddingSize)
    new Sequential()
      .add(wordVectors)
      .add(SplitTable(1, 2))
      .add(ConcatTable()
        .add(Sequential()
          .add(NarrowTable(2, -1))
          .add(JoinTable(1, 2)))
        .add(SelectTable(1)))
      .add(MM(transA = false, transB = true))
      .add(Sigmoid())
  }

  def buildVocab(words: RDD[String]): Unit = {
    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= minCount)
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
    var idx = 0
    var powerProb = wordProb(wordId)
    while (idx < powerSum) {
      powerUnigram(idx) = wordId
      if (idx / powerSum.toDouble > powerProb) {
        wordId += 1
        powerProb += wordProb(wordId)
      }

      if (wordId >= vocab.length - 1) {
        wordId = vocab.length - 2
      }

      idx += 1
    }
  }

  /**
   * Sample negative contexts from power unigram distribution,
   * and conbine them with given word and context together as a training sample
   *
   * @param powerUnigram the power unigram distribution
   * @param word given word
   * @param context given context
   * @return a one-dimension tensor, whose first elmement is the given word,
   *         second element is the given context, and the rest elements are the
   *         sampled negative contexts.
   */
  def sampleContexts(
    powerUnigram: Array[Int],
    word: Int,
    context: Int): Tensor[Float] = {
    val contexts = Tensor(params.numNegSamples + 2)
    contexts.setValue(1, word)
    contexts.setValue(2, context)
    var i = 3
    while (i <= params.numNegSamples + 2) {
      // Sample a negative context
      val negContext = powerUnigram(RNG.uniform(0, powerSum).toInt)
      if (context != negContext) {
        contexts.setValue(i, negContext)
        i += 1
      }
    }
    contexts.view(1, contexts.size(1))
  }

  def transformToBatch(): Transformer[Seq[String], MiniBatch[Float]] = {
    (WordsToIds(word2Id, params.maxSentenceLength)
      -> Subsampling(params.subsample, trainWordsCount, vocab)
      -> SentenceToSamples(powerUnigram, params.numNegSamples, params.windowSize)
      -> SampleToBatch(16))
  }

  def fit[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    val words = dataset.flatMap(x => x)

    buildVocab(words)

    buildNegSampleDistribution()
  }

  /**
   * Normalize the each word vector
   */
  def normalizeWordVectors(): Unit =
    wordVectors.weight.set(normalizeMatrix(wordVectors.weight))

  /**
   * Normalize rows of a matrix
   *
   * @param matrix given matrix
   * @return nomalized matrix
   */
  def normalizeMatrix(matrix: Tensor[Float]): Tensor[Float] = {
    val size = matrix.size()
    val normMatrix = Tensor(size).zero()

    var i = 1
    while (i < size(0)) {
      val norm = matrix.apply(i).norm(1)
      var j = 1
      while (j < size(1)) {
        normMatrix.setValue(i, j, matrix(Array(i, j)) / norm)
        j += 1
      }
      i += 1
    }

    normMatrix
  }

  /**
   * Return the k-nearest words to a word or a vector based on cosine similarity
   * @param numSimilarWord Output the number of most similar words given a
   *                       input during prediction1
   */
  def getSimilarWords(
    words: Array[String],
    numSimilarWord: Int): Array[Array[String]] = {
    words.map(word => {
      if (!word2Id.contains(word)) {
        log.info(s"$word does not exist in vocabulary.")
        null
      } else {
        val vector = wordVectors.weight(word2Id(word))
        val similarity = Tensor().mv(wordVectors.weight, vector)
        argsort(similarity.toBreezeVector())
          .reverse
          .toArray
          .map(id => vocab(id).word)
      }
    }).filter(_ != null)
  }

  /**
   * Transform a sequence of words to its corresponding ids, and filter the length
   * of each set of words less than `maxSentenceLength`
   *
   * @param word2Id A Map from [[String]] of word to the index of the word in dictionary
   * @param maxSentenceLength The maximum sentence length
   */
  case class WordsToIds(
    word2Id: mutable.HashMap[String, Int],
    maxSentenceLength: Int)
    extends Transformer[Seq[String], Seq[Int]] {
    val sentence = mutable.ArrayBuilder.make[Int]

    override def apply(prev: Iterator[Seq[String]]): Iterator[Seq[Int]] =
      prev.map(words => {
        var sentenceLength = 0
        while (sentenceLength < words.length &&
          sentenceLength < maxSentenceLength) {
          val word = word2Id.get(words(sentenceLength))
          word match {
            case Some(w) =>
              sentence += w + 1
              sentenceLength += 1
            case None =>
          }
          sentenceLength += 1
        }
        sentence.result()
      })
  }

  /**
   * Subsampling of Frequent Words.
   * This transformer will use a simple subsampling approach:
   *    each word w in training set is discarded with a probability computed
   *    by the formula:
   *
   *    P(w_i) = (f(w_i) - t) / f(w_i) - sqrt(t / (f(w_i)))
   *
   *    where `t` is the threshold
   *
   * @param t threshould
   * @param trainWordsCount the total number of words in training set
   * @param vocab vocabulary of the training words
   */
  case class Subsampling(
    t: Double,
    trainWordsCount: Long,
    vocab: Array[WordCount])
    extends Transformer[Seq[Int], Seq[Int]] {
    val sentence = mutable.ArrayBuilder.make[Int]

    override def apply(prev: Iterator[Seq[Int]]): Iterator[Seq[Int]] =
      if (t > 0) {
        prev.map(words => {
          words.map(word => {
            val ran = (math.sqrt(vocab(word).count / (t * trainWordsCount)) + 1) *
              (t * trainWordsCount) / vocab(word).count

            if (ran < RNG.uniform(0, 1)) {
              -1
            } else {
              word
            }
          }).filter(_ != -1)
        })
      } else {
        prev
      }
  }

  /**
   * Transform sentence represented as word ids to MiniBatch
   * Each sensence is represented as a batch
   *
   * @param powerUnigram the power unigram distribution
   * @param numNegSamples Number of negative samples per example.
   * @param window The number of words to predict to the left
   *                   and right of the target word.
   */
  case class SentenceToSamples(
    powerUnigram: Array[Int],
    numNegSamples: Int,
    window: Int)
    extends Transformer[Seq[Int], Sample[Float]] {
    override def apply(prev: Iterator[Seq[Int]]): Iterator[Sample[Float]] = {
      prev.flatMap(sentence => {
        val samples = mutable.ArrayBuilder.make[Sample[Float]]
        sentence.zipWithIndex.foreach {
          case (word, i) =>
            val reducedWindow = RNG.uniform(1, window + 1).toInt
            var j = i - reducedWindow
            j = if (j < 0) 0 else j

            while (j <= i + reducedWindow && j < sentence.length) {
              if (j != i) {
                val context = sentence(j)
                val sample = sampleContexts(powerUnigram, word, context)
                samples += new Sample(sample, Tensor(T(1f)))
              }
              j += 1
            }
        }
        samples.result()
      })
    }
  }


//  /**
//   * Batching samples into mini-batch
//   * @param batchSize The desired mini-batch size.
//   */
//  case class SampleToBatch(batchSize: Int)
//    extends Transformer[Seq[Sample[Float]], MiniBatch[Float]] {
//    override def apply(prev: Iterator[Seq[Sample[Float]]]): Iterator[MiniBatch[Float]] = {
//      new Iterator[MiniBatch[Float]] {
//        private val featureTensor: Tensor[Float] = Tensor[Float]()
//        private val labelTensor: Tensor[Float] = Tensor[Float]()
//        private var featureData: Array[Float] = _
//        private var labelData: Array[Float] = _
//
//        override def hasNext: Boolean = prev.hasNext
//
//        override def next(): MiniBatch[Float] = {
//          var i = 0
//          while (i < batchSize && prev.hasNext) {
//            val sample = prev.next()
//            if (featureData == null) {
//              featureData = new Array[Float](batchSize * sampleSize)
//              labelData = new Array[Float](batchSize)
//            }
//            Array.copy(sample._1.flatten, 0,
//              featureData, i * sampleSize, sampleSize)
//            labelData(i) = sample._2
//            i += 1
//          }
//          featureTensor.set(Storage[Float](featureData),
//            storageOffset = 1, sizes = Array(i) ++ sampleShape)
//          labelTensor.set(Storage[Float](labelData),
//            storageOffset = 1, sizes = Array(i))
//          MiniBatch(featureTensor.transpose(2, 3), labelTensor)
//        }
//      }
//    }
//  }
}
