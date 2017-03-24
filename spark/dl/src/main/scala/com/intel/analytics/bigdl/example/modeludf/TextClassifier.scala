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

package com.intel.analytics.bigdl.example.modeludf

import java.io.File
import java.util

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.example.textclassification.SimpleTokenizer._
import com.intel.analytics.bigdl.example.textclassification.{SimpleTokenizer, WordMeta}
import com.intel.analytics.bigdl.example.modeludf.Options.TextClassificationParams
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, _}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.Source
import scala.language.existentials


  /**
  * This example use a (pre-trained GloVe embedding) to convert word to vector,
  * and uses it to train a text classification model on the 20 Newsgroup dataset
  * with 20 different categories. This model can achieve around 90% accuracy after
  * 2 epoches training.
  */
class TextClassifier(param: TextClassificationParams) extends Serializable {
  private val log: Logger = LoggerFactory.getLogger(this.getClass)
  private val gloveDir = s"${param.baseDir}/glove.6B/"
  private val textDataDir = s"${param.baseDir}/20_newsgroup/"
  private val testDataDir = s"${param.testDir}"
  private var classNum = -1

    /**
    * Load the pre-trained word2Vec
    *
    * @return A map from word to vector
    */
  private def buildWord2Vec(word2Meta: Map[String, WordMeta]): Map[Float, Array[Float]] = {
    log.info("Indexing word vectors.")
    val preWord2Vec = MMap[Float, Array[Float]]()
    val filename = s"$gloveDir/glove.6B.100d.txt"
    for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      if (word2Meta.contains(word)) {
        val coefs = values.slice(1, values.length).map(_.toFloat)
        preWord2Vec.put(word2Meta(word).index.toFloat, coefs)
      }
    }
    log.info(s"Found ${preWord2Vec.size} word vectors.")
    preWord2Vec.toMap
  }

    /**
    * Load the pre-trained word2Vec
    *
    * @return A map from word to vector
    */
  def buildWord2VecMap(): Map[String, Array[Float]] = {
    log.info("Indexing word vectors.")
    val preWord2Vec = MMap[String, Array[Float]]()
    val filename = s"$gloveDir/glove.6B.100d.txt"
    for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toFloat)
      preWord2Vec.put(word, coefs)

    }
    log.info(s"Found ${preWord2Vec.size} word vectors.")
    preWord2Vec.toMap
  }

    /**
    * Load the test data from the given baseDir
    *
    * @return An array of sample
    */
  def loadTestData(): Array[(String, String)] = {
    // category is a string name, label is it's index
    //     val categoryToLabel = new util.HashMap[String, Int]()
    val fileList = new File(testDataDir).listFiles()
      .filter(_.isFile).filter(_.getName.forall(Character.isDigit)).sorted

    val testData = fileList.map { file => {
      val fileName = file.getName
      val source = Source.fromFile(file, "ISO-8859-1")
      val text = try source.getLines().toList.mkString("\n") finally source.close()
      (fileName, text)
    }
    }
    testData
  }

    /**
    * Load the training data from the given baseDir
    *
    * @return An array of sample
    */
  private def loadRawData(): ArrayBuffer[(String, String, Float)] = {
    val fileNames = ArrayBuffer[String]()
    val texts = ArrayBuffer[String]()
    val labels = ArrayBuffer[Float]()
    // category is a string name, label is it's index
    val categoryToLabel = new util.HashMap[String, Int]()
    val categoryPathList = new File(textDataDir).listFiles().filter(_.isDirectory).toList.sorted

    categoryPathList.foreach { categoryPath =>
      val label_id = categoryToLabel.size() + 1 // one-base index
      categoryToLabel.put(categoryPath.getName, label_id)
      val textFiles = categoryPath.listFiles()
        .filter(_.isFile).filter(_.getName.forall(Character.isDigit)).sorted
      textFiles.foreach { file =>
        val fileName = file.getName
        val source = Source.fromFile(file, "ISO-8859-1")
        val text = try source.getLines().toList.mkString("\n") finally source.close()
        fileNames.append(fileName)
        texts.append(text)
        labels.append(label_id)
      }
    }
    this.classNum = labels.toSet.size
    log.info(s"Found ${texts.length} texts.")
    log.info(s"Found $classNum classes")
    fileNames.zip(texts).zip(labels).map {
      case ((filename, text), label) => (filename, text, label)
    }
  }

    /**
    * Go through the whole data set to gather some meta info for the tokens.
    * Tokens would be discarded if the frequency ranking is less then maxWordsNum
    */
  def analyzeTexts(dataRdd: RDD[(String, String, Float)])
  : (Map[String, WordMeta], Map[Float, Array[Float]]) = {
    // Remove the top 10 words roughly, you might want to fine tuning this.
    val frequencies = dataRdd.flatMap { case (_: String, text: String, _: Float) =>
      SimpleTokenizer.toTokens(text)
    }.map(word => (word, 1)).reduceByKey(_ + _)
      .sortBy(-_._2).collect().slice(10, param.maxWordsNum)

    val indexes = Range(1, frequencies.length)
    val word2Meta = frequencies.zip(indexes).map { item =>
      (item._1._1, WordMeta(item._1._2, item._2))
    }.toMap
    (word2Meta, buildWord2Vec(word2Meta))
  }

  // TODO: Replace SpatialConv and SpatialMaxPolling with 1D implementation
  def buildModel(classNum: Int): Sequential[Float] = {
    val model = Sequential[Float]()

    model.add(Reshape(Array(param.embeddingDim, 1, param.maxSequenceLength)))

    model.add(SpatialConvolution(param.embeddingDim, 128, 5, 1))
    model.add(ReLU())

    model.add(SpatialMaxPooling(5, 1, 5, 1))

    model.add(SpatialConvolution(128, 128, 5, 1))
    model.add(ReLU())

    model.add(SpatialMaxPooling(5, 1, 5, 1))

    model.add(SpatialConvolution(128, 128, 5, 1))
    model.add(ReLU())

    model.add(SpatialMaxPooling(35, 1, 35, 1))

    model.add(Reshape(Array(128)))
    model.add(Linear(128, 100))
    model.add(Linear(100, classNum))
    model.add(LogSoftMax())
    model
  }


    /**
    * Create train and val RDDs from input
    */
  def getData(sc: SparkContext):
  Array[RDD[(String, Array[Array[Float]], Float)]] = {

    val sequenceLen = param.maxSequenceLength
    val embeddingDim = param.embeddingDim
    val trainingSplit = param.trainingSplit
    // For large dataset, you might want to get such RDD[(String, Float)] from HDFS
    val dataRdd = sc.parallelize(loadRawData(), param.partitionNum)
    val (word2Meta, word2Vec) = analyzeTexts(dataRdd)
    val word2MetaBC = sc.broadcast(word2Meta)
    val word2VecBC = sc.broadcast(word2Vec)
    val vectorizedRdd = dataRdd
      .map { case (filename, text, label) => (filename, toTokens(text, word2MetaBC.value), label) }
      .map { case (filename, tokens, label) => (filename, shaping(tokens, sequenceLen), label) }
      .map { case (filename, tokens, label) => (filename, vectorization(
        tokens, embeddingDim, word2VecBC.value), label)
      }

    vectorizedRdd.randomSplit(
      Array(trainingSplit, 1 - trainingSplit))

  }

    /**
    * Train the text classification model with train and val RDDs
    */
  def train(sc: SparkContext, rdds: Array[RDD[(String, Array[Array[Float]], Float)]])
  : Module[Float] = {

    // create rdd from input directory
    val trainingRDD = rdds(0).map {
      case (_, text, label) => (text, label)
    }.map { case (input: Array[Array[Float]], label: Float) =>
      Sample(
        featureTensor = Tensor(input.flatten, Array(param.maxSequenceLength, param.embeddingDim))
          .transpose(1, 2).contiguous(),
        labelTensor = Tensor(Array(label), Array(1)))
    }

    val valRDD = rdds(1).map {
      case (_, text, label) => (text, label)
    }.map { case (input: Array[Array[Float]], label: Float) =>
      Sample(
        featureTensor = Tensor(input.flatten, Array(param.maxSequenceLength, param.embeddingDim))
          .transpose(1, 2).contiguous(),
        labelTensor = Tensor(Array(label), Array(1)))
    }

    // train
    val optimizer = Optimizer(
      model = buildModel(classNum),
      sampleRDD = trainingRDD,
      criterion = new ClassNLLCriterion[Float](),
      batchSize = param.batchSize
    )

    val state = T("learningRate" -> 0.01, "learningRateDecay" -> 0.0002)
    optimizer
      .setState(state)
      .setOptimMethod(new Adagrad())
      .setValidation(Trigger.everyEpoch, valRDD, Array(new Top1Accuracy[Float]), param.batchSize)
      .setEndWhen(Trigger.maxEpoch(1))
      .optimize()
  }
}
