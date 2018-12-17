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

package com.intel.analytics.bigdl.example.udfpredictor

import java.io.{File, InputStream, PrintWriter}

import com.intel.analytics.bigdl.example.utils.WordMeta
import com.intel.analytics.bigdl.example.utils.TextClassifier
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.Module
import org.apache.spark.SparkContext

import scala.io.Source
import scopt.OptionParser


object Utils {

  type Model = AbstractModule[Activity, Activity, Float]
  type Word2Meta = Map[String, WordMeta]
  type Word2Index = Map[String, Int]
  type Word2Vec = Map[Float, Array[Float]]
  type SampleShape = Array[Int]
  type TFP = TextClassificationUDFParams

  case class Sample(filename: String, text: String)

  private var textClassification: TextClassifier = null

  def getTextClassifier(param: TextClassificationUDFParams): TextClassifier = {
    if (textClassification == null) {
      textClassification = new TextClassifier(param)
    }
    textClassification
  }

  def getModel(sc: SparkContext, param: TFP): (Model, Option[Word2Meta],
    Option[Word2Vec], SampleShape) = {
    val textClassification = getTextClassifier(param)
    if (param.modelPath.isDefined) {
      (Module.load[Float](param.modelPath.get),
        None,
        None,
        Array(param.maxSequenceLength, param.embeddingDim))
    } else {
      // get train and validation rdds
      val (rdds, word2Meta, word2Vec) = textClassification.getData(sc)
      // save word2Meta for later generate vectors
      val word2Index = word2Meta.mapValues[Int]((wordMeta: WordMeta) => wordMeta.index)
      sc.parallelize(word2Index.toSeq).saveAsTextFile(s"${param.baseDir}/word2Meta.txt")
      // train
      val trainedModel = textClassification.trainFromData(sc, rdds)
      // after training, save model
      if (param.checkpoint.isDefined) {
        trainedModel.save(s"${param.checkpoint.get}/model.1", overWrite = true)
      }

      (trainedModel.evaluate(),
        Some(word2Meta),
        Some(word2Vec),
        Array(param.maxSequenceLength, param.embeddingDim))
    }
  }

  def getWord2Vec(word2Index: Map[String, Int]): Map[Float, Array[Float]] = {
    val word2Vec = textClassification.buildWord2VecWithIndex(word2Index)
    word2Vec
  }

  def genUdf(sc: SparkContext,
             model: Model,
             sampleShape: Array[Int],
             word2Index: Word2Index,
             word2Vec: Word2Vec)
            (implicit ev: TensorNumeric[Float]): (String) => Int = {

    val broadcastModel = ModelBroadcast[Float]().broadcast(sc, model)
    val word2IndexBC = sc.broadcast(word2Index)
    val word2VecBC = sc.broadcast(word2Vec)

    val udf = (text: String) => {
      val sequenceLen = sampleShape(0)
      val embeddingDim = sampleShape(1)
      val word2Meta = word2IndexBC.value
      val word2Vec = word2VecBC.value
      // first to tokens
      val tokens = text.replaceAll("[^a-zA-Z]", " ")
        .toLowerCase().split("\\s+").filter(_.length > 2).map { word: String =>
        if (word2Meta.contains(word)) {
          Some(word2Meta(word).toFloat)
        } else {
          None
        }
      }.flatten

      // shaping
      val paddedTokens = if (tokens.length > sequenceLen) {
        tokens.slice(tokens.length - sequenceLen, tokens.length)
      } else {
        tokens ++ Array.fill[Float](sequenceLen - tokens.length)(0)
      }

      val data = paddedTokens.map { word: Float =>
        if (word2Vec.contains(word)) {
          word2Vec(word)
        } else {
          // Treat it as zeros if cannot be found from pre-trained word2Vec
          Array.fill[Float](embeddingDim)(0)
        }
      }.flatten

      val featureTensor: Tensor[Float] = Tensor[Float]()
      var featureData: Array[Float] = null
      val sampleSize = sampleShape.product
      val localModel = broadcastModel.value()

      // create tensor from input column
      if (featureData == null) {
        featureData = new Array[Float](1 * sampleSize)
      }
      Array.copy(data.map(ev.fromType(_)), 0,
        featureData, 0, sampleSize)
      featureTensor.set(Storage[Float](featureData), sizes = Array(1) ++ sampleShape)
      val tensorBuffer = featureTensor.transpose(2, 3)

      // predict
      val output = localModel.forward(tensorBuffer).toTensor[Float]
      val predict = if (output.dim == 2) {
        output.max(2)._2.squeeze().storage().array()
      } else if (output.dim == 1) {
        output.max(1)._2.squeeze().storage().array()
      } else {
        throw new IllegalArgumentException
      }
      ev.toType[Int](predict(0))
    }

    udf
  }

  def loadTestData(testDir: String): IndexedSeq[Sample] = {
    val fileList = new File(testDir).listFiles()
      .filter(_.isFile).filter(_.getName.forall(Character.isDigit)).sorted

    val testData = fileList.map { file => {
      val fileName = file.getName
      val source = Source.fromFile(file, "ISO-8859-1")
      val text = try source.getLines().toList.mkString("\n") finally source.close()
      Sample(fileName, text)
    }
    }
    testData
  }

  def getResourcePath(resource: String): String = {
    val stream: InputStream = getClass.getResourceAsStream(resource)
    val lines = scala.io.Source.fromInputStream(stream).mkString
    val file = File.createTempFile(resource, "")
    val pw = new PrintWriter(file)
    pw.write(lines)
    pw.close()
    file.getAbsolutePath
  }

  val localParser = new OptionParser[TextClassificationUDFParams]("BigDL Example") {
    opt[String]('b', "baseDir")
      .text("Base dir containing the training and word2Vec data")
      .action((x, c) => c.copy(baseDir = x))
    opt[String]('p', "partitionNum")
      .text("you may want to tune the partitionNum if run into spark mode")
      .action((x, c) => c.copy(partitionNum = x.toInt))
    opt[String]('s', "maxSequenceLength")
      .text("maxSequenceLength")
      .action((x, c) => c.copy(maxSequenceLength = x.toInt))
    opt[String]('w', "maxWordsNum")
      .text("maxWordsNum")
      .action((x, c) => c.copy(maxWordsNum = x.toInt))
    opt[String]('l', "trainingSplit")
      .text("trainingSplit")
      .action((x, c) => c.copy(trainingSplit = x.toDouble))
    opt[String]('z', "batchSize")
      .text("batchSize")
      .action((x, c) => c.copy(batchSize = x.toInt))
    opt[String]("modelPath")
      .text("where to load the model")
      .action((x, c) => c.copy(modelPath = Some(x)))
    opt[String]("checkpoint")
      .text("where to load the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[String]('f', "dataDir")
      .text("Text dir containing the text data")
      .action((x, c) => c.copy(testDir = x))
  }
}
