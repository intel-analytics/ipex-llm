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
import breeze.numerics.sqrt
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, Transformer}
import com.intel.analytics.bigdl.models.embedding.Utils.Word2VecConfig
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
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

/**
 * This version of word2Vec implementation using
 *    skip-gram + negative sampling and sub-sampling.
 *
 * For more details of the theory, please refer to the paper:
 *   Distributed Representations of Words and Phrases and their Compositionality
 *
 * @param params word2vec configuration parameters
 */
@SerialVersionUID(- 4753052739687459443L)
class Word2Vec(val params: Word2VecConfig)
  extends Serializable {
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  private var trainWordsCount = 0L
  private var vocabSize = 0
  private val minCount = params.minCount
  private var vocab: Array[WordCount] = _
  private var word2Id = mutable.HashMap.empty[String, Int]

  final val tableSize = 1e8.toInt
  private val powerUnigram: Array[Int] = new Array(tableSize)

  // embedding the word vectors in the LookupTable
  var wordVectors: LookupTable[Float] = _
  var contextVectors: LookupTable[Float] = _

  /**
   * Generate the training model of word2vec
   */
  def getModel: Module[Float] = {
    wordVectors = LookupTable(vocabSize, params.embeddingSize)
    wordVectors.weight.apply1(x => RNG.uniform(-0.5, 0.5).toFloat / params.embeddingSize)
    contextVectors = LookupTable(vocabSize, params.embeddingSize)
    contextVectors.reset(1.0 / sqrt(params.embeddingSize))

    new Sequential()
      .add(ConcatTable()
        .add(Narrow(2, 2, params.numNegSamples + 1))
        .add(Narrow(2, 1, 1)))
      .add(ParallelTable()
        .add(contextVectors)
        .add(wordVectors))
      .add(MM(transA = false, transB = true))
      .add(Sigmoid())
  }

  /**
   * Prepare the data including build the dictionary and initialize
   * the negative sampling distribution
   *
   * @param dataset text data set in [[RDD]] format
   *                each element of RDD is a sentence containing the words
   *                in iterable format
   */
  def initialize[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    val words = dataset.flatMap(x => x)

    log.info(s"${words.count()} words and ${dataset.count()} sentences processed")

    buildVocab(words)

    buildNegSampleDistribution()
  }

  /**
   * Build the vocabulary frequency, word2Id from input
   *
   * @param words the given words, each element in the RDD is a word
   */
  def buildVocab(words: RDD[String]): Unit = {
    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= minCount)
      .map(x => WordCount(x._1, x._2))
      .collect()

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
    val powerSum =
      vocab.foldLeft(0.0)((sum, wordCount) => sum + math.pow(wordCount.count, params.alpha))

    def wordProb(id: Int) = math.pow(vocab(id).count, params.alpha) / powerSum

    var wordId = 0
    var idx = 0
    var powerProb = wordProb(wordId)
    while (idx < tableSize) {
      powerUnigram(idx) = wordId
      if (idx / tableSize.toDouble > powerProb) {
        wordId += 1
        powerProb += wordProb(wordId)
      }

      if (wordId >= vocab.length) {
        wordId = vocab.length - 1
      }

      idx += 1
    }
  }

  /**
   * Pre-process the text, generate training data and
   * transform it to the train samples in [[MiniBatch]] format
   *
   * @return a transformer transforms the input words to Mini-batch
   */
  def generateTrainingData(): Transformer[Seq[String], Sample[Float]] = {
    (WordsToIds(word2Id, params.maxSentenceLength)
      -> SubSampling(params.subsample, trainWordsCount, vocab)
      -> GenerateSamplesBySkipGram(powerUnigram, params.numNegSamples, params.windowSize))
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
    override def apply(prev: Iterator[Seq[String]]): Iterator[Seq[Int]] = {
      prev.map(words => {
        val sentence = mutable.ArrayBuilder.make[Int]
        var sentenceLength = 0
        var i = 0

        while (i < words.length && sentenceLength < maxSentenceLength) {
          val word = word2Id.get(words(i))
          word match {
            case Some(w) =>
              sentence += w
              sentenceLength += 1
            case None =>
          }
          i += 1
        }
        sentence.result()
      })
    }
  }

  /**
   * Sub-sampling of Frequent Words.
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
  case class SubSampling(
    t: Double,
    trainWordsCount: Long,
    vocab: Array[WordCount])
    extends Transformer[Seq[Int], Seq[Int]] {
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
   * Sample negative contexts from power unigram distribution,
   * and combine them with given word and context together as a training sample
   *
   * @param powerUnigram the power unigram distribution
   * @param word given word
   * @param context given context
   * @return a one-dimension tensor, whose first element is the given word,
   *         second element is the given context, and the rest elements are the
   *         sampled negative contexts.
   */
  def sampleContexts(
    powerUnigram: Array[Int],
    word: Int,
    context: Int): Tensor[Float] = {
    val contexts = Tensor(params.numNegSamples + 2)
    contexts.setValue(1, word + 1)
    contexts.setValue(2, context + 1)
    var i = 3

    while (i <= params.numNegSamples + 2) {
      // Sample a negative context
      val negContext = powerUnigram(RNG.uniform(0, tableSize).toInt)
      if (context != negContext) {
        contexts.setValue(i, negContext + 1)
        i += 1
      }
    }
    contexts
  }

  /**
   * Generate training samples by skip-gram algorithm
   *
   * @param powerUnigram the power unigram distribution
   * @param numNegSamples Number of negative samples per example.
   * @param window The number of words to predict to the left
   *                   and right of the target word.
   */
  case class GenerateSamplesBySkipGram(
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
                val label = Tensor(1 + numNegSamples)
                              .zero()
                              .setValue(1, 1)
                samples += new Sample(sample, label)
              }
              j += 1
            }
        }
        samples.result()
      })
    }
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
   * @return normalized matrix
   */
  def normalizeMatrix(matrix: Tensor[Float]): Tensor[Float] = {
    val size = matrix.size()
    val normMatrix = Tensor(size).zero()

    var i = 1
    while (i <= size(0)) {
      val norm = matrix(i).norm(2)
      var j = 1
      while (j <= size(1)) {
        normMatrix.setValue(i, j, matrix(Array(i, j)) / norm)
        j += 1
      }
      i += 1
    }

    normMatrix
  }

  /**
   * Print the k-nearest words to a word or a vector based on cosine similarity
   * @param numSimilarWord print the number of the most similar words
   */
  def printSimilarWords(
    words: Array[String],
    numSimilarWord: Int): Unit = {
    val simWords = getSimilarWords(words, numSimilarWord)
    log.info(simWords
      .map(e => e.mkString(", "))
      .mkString("\n"))
  }

  /**
   * Return the k-nearest words to a set of words based on cosine similarity
   *
   * @param numSimilarWord return the number of the most similar words given a
   *                       input during prediction
   * @return ranked most similar words, with corresponding similarity
   */
  def getSimilarWords(
    words: Array[String],
    numSimilarWord: Int): Array[Array[(String, Float)]] = {
    words.map(word => {
      val lower = word.toLowerCase()
      if (!word2Id.contains(lower)) {
        log.info(s"$lower does not exist in vocabulary.")
        null
      } else {
        val vector = getVectorByString(lower)
        getSimilarWordsGivenAVector(vector, numSimilarWord)
      }
    }).filter(_ != null)
  }

  /**
   * Return the k-nearest words to a vector based on cosine similarity
   *
   * @param vector an input vector
   * @param numSimilarWord return the number of the most similar words given a
   *                       input during prediction
   * @return ranked most similar words, with corresponding similarity
   */
  def getSimilarWordsGivenAVector(
    vector: Tensor[Float],
    numSimilarWord: Int): Array[(String, Float)] = {
    val similarity = Tensor(vocabSize)
      .mv(wordVectors.weight, vector)
      .toBreezeVector()

    implicit val ordering = Ordering.Float.reverse
    argsort(similarity)
      .take(numSimilarWord)
      .toArray
      .map(id => (vocab(id).word, similarity(id)))
  }

  /**
   * Given a word in [[String]] format, return the corresponding word vector
   * @param word word in [[String]] format
   * @return
   */
  def getVectorByString(word: String): Tensor[Float] = {
    getVectorById(word2Id(word))
  }

  /**
   * Given a word id in [[Int]] format, return the corresponding word vector
   * @param id word id in [[Int]] format
   * @return
   */
  def getVectorById(id: Int): Tensor[Float] = {
    wordVectors.weight(id + 1)
  }

  def printWordAnalogy(
    words: Array[String],
    numSimilarWord: Int): Unit = {
    val simWords = getWordAnalogy(words, numSimilarWord)
    log.info(simWords.mkString(", ") + "\n")
  }

  /**
   * Given three words in `words`, return the most similar analogy (word4)
   * to match the relationship between (word1, word2) and (word3, word4).
   *
   * For example:
   *    Given ["Beijing", "China", "London"], expected result should be
   *    [England, ...]
   */
  def getWordAnalogy(
    words: Array[String],
    numSimilarWord: Int): Array[(String, Float)] = {
    if (words.length < 3) {
      println(s"Only ${words.length} words were entered.." +
        s" three words are needed at the input to perform the calculation")
    }

    val vectors = words.map(word => {
      val lower = word.toLowerCase()
      getVectorByString(lower)
    })

    val vector = vectors(0) - vectors(1) + vectors(2)
    getSimilarWordsGivenAVector(vector, numSimilarWord)
  }
}
