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

package com.intel.analytics.bigdl.example.modeludf

import java.io.{File, InputStream, PrintWriter}

import com.intel.analytics.bigdl.example.modeludf.FileProducer.Sample
import com.intel.analytics.bigdl.example.modeludf.Options.TextClassificationParams
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.SparkContext

import scala.io.Source


object Utils {

  type Model = AbstractModule[Activity, Activity, Float]
  type Word2Vec = Map[String, Array[Float]]
  type SampleShape = Array[Int]
  type TFP = TextClassificationParams



  def getModel(sc: SparkContext, param: TFP): (Model, Word2Vec, SampleShape) = {
    val textClassification = new TextClassifier(param)
    val model = param.modelPath.map { path =>
      Module.load[Float](path)
    }.getOrElse {
      // get train and validation rdds
      val rdds = textClassification.getData(sc)
      // train
      val trainedModel = textClassification.train(sc, rdds)
      // after training, save model
      if (param.checkpoint.isDefined) {
        trainedModel.save(s"${param.checkpoint.get}/model.1", overWrite = true)
      } else {
        trainedModel
      }
    }
    (model.evaluate(),
      textClassification.buildWord2VecMap(),
      Array(param.maxSequenceLength, param.embeddingDim))
  }

  def genUdf(sc: SparkContext,
             model: Model,
             sampleShape: Array[Int],
             word2Vec: Word2Vec)
            (implicit ev: TensorNumeric[Float]): (String) => Int = {

    val broadcastModel = sc.broadcast(model)
    val broadcastWord2Vec = sc.broadcast(word2Vec)

    val udf = (text: String) => {
      val sequenceLen = sampleShape(0)
      val embeddingDim = sampleShape(1)
      // first to tokens
      val tokens = text.replaceAll("[^a-zA-Z]", " ")
        .toLowerCase().split("\\s+").filter(_.length > 2)
      // shaping
      val paddedTokens = if (tokens.length > sequenceLen) {
        tokens.slice(tokens.length - sequenceLen, tokens.length)
      } else {
        tokens ++ Array.fill[String](sequenceLen - tokens.length)("invalidword")
      }
      // to vectors
      val word2Vec = broadcastWord2Vec.value
        .withDefaultValue(Array.fill[Float](embeddingDim)(0))

      val data = paddedTokens.flatMap(word2Vec.apply)

      val featureTensor: Tensor[Float] = Tensor[Float]()
      var featureData: Array[Float] = null
      val sampleSize = sampleShape.product
      val localModel = broadcastModel.value

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

  /**
    * Get resource file path
    *
    * @param resource
    * @return file path
    */
  def getResourcePath(resource: String): String = {
    val stream: InputStream = getClass.getResourceAsStream(resource)
    val lines = scala.io.Source.fromInputStream(stream).mkString
    val file = File.createTempFile(resource, "")
    val pw = new PrintWriter(file)
    pw.write(lines)
    pw.close()
    file.getAbsolutePath
  }

}
