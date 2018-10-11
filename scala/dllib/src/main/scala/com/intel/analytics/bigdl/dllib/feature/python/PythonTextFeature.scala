/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.feature.python

import java.util.{List => JList, Map => JMap}

import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL, Sample}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dataset.{Sample => JSample}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.feature.text.TruncMode.TruncMode
import com.intel.analytics.zoo.feature.text.{DistributedTextSet, _}
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonTextFeature {

  def ofFloat(): PythonTextFeature[Float] = new PythonTextFeature[Float]()

  def ofDouble(): PythonTextFeature[Double] = new PythonTextFeature[Double]()
}

class PythonTextFeature[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def createTextFeature(text: String): TextFeature = {
    TextFeature(text)
  }

  def createTextFeature(text: String, label: Int): TextFeature = {
    TextFeature(text, label)
  }

  def textFeatureGetText(feature: TextFeature): String = {
    feature.getText
  }

  def textFeatureGetLabel(feature: TextFeature): Int = {
    feature.getLabel
  }

  def textFeatureHasLabel(feature: TextFeature): Boolean = {
    feature.hasLabel
  }

  def textFeatureSetLabel(feature: TextFeature, label: Int): TextFeature = {
    feature.setLabel(label)
  }

  def textFeatureGetKeys(feature: TextFeature): JList[String] = {
    feature.keys().toList.asJava
  }

  def textFeatureGetTokens(feature: TextFeature): JList[String] = {
    val tokens = feature.getTokens
    if (tokens != null ) tokens.toList.asJava else null
  }

  def textFeatureGetSample(feature: TextFeature): Sample = {
    val sample = feature.getSample
    if (sample != null) toPySample(sample.asInstanceOf[JSample[T]]) else null
  }

  def transformTextFeature(
      transformer: TextTransformer,
      feature: TextFeature): TextFeature = {
    transformer.transform(feature)
  }

  def createTokenizer(): Tokenizer = {
    Tokenizer()
  }

  def createNormalizer(): Normalizer = {
    Normalizer()
  }

  def createWordIndexer(map: JMap[String, Int], replaceElement: Int): WordIndexer = {
    WordIndexer(map.asScala.toMap, replaceElement)
  }

  def createSequenceShaper(
      len: Int,
      truncMode: String,
      padElement: String): SequenceShaper = {
    SequenceShaper(len, toScalaTruncMode(truncMode), padElement)
  }

  private def toScalaTruncMode(str: String): TruncMode = {
    str.toLowerCase() match {
      case "pre" => TruncMode.pre
      case "post" => TruncMode.post
    }
  }

  def createTextFeatureToSample(): TextFeatureToSample = {
    TextFeatureToSample()
  }

  def createLocalTextSet(texts: JList[String], labels: JList[Int]): LocalTextSet = {
    require(texts != null, "texts of a TextSet can't be null")
    val features = if (labels != null) {
      require(texts.size() == labels.size(), "texts and labels of a TextSet " +
        "should have the same size")
      texts.asScala.toArray[String].zip(labels.asScala.toArray[Int]).map{feature =>
        createTextFeature(feature._1, feature._2)
      }
    }
    else {
      texts.asScala.toArray.map(createTextFeature)
    }
    TextSet.array(features)
  }

  def createDistributedTextSet(
      texts: JavaRDD[String],
      labels: JavaRDD[Int]): DistributedTextSet = {
    require(texts != null, "texts of a TextSet can't be null")
    val features = if (labels != null) {
      texts.rdd.zip(labels.rdd).map{feature =>
        createTextFeature(feature._1, feature._2)
      }
    }
    else {
      texts.rdd.map(createTextFeature)
    }
    TextSet.rdd(features)
  }

  def readTextSet(path: String, sc: JavaSparkContext, minPartitions: Int): TextSet = {
    if (sc == null) {
      TextSet.read(path, null, minPartitions)
    }
    else {
      TextSet.read(path, sc.sc, minPartitions)
    }
  }

  def textSetGetWordIndex(textSet: TextSet): JMap[String, Int] = {
    val res = textSet.getWordIndex
    if (res == null) null else res.asJava
  }

  def textSetGenerateWordIndexMap(
      textSet: TextSet,
      removeTopN: Int = 0,
      maxWordsNum: Int = -1): JMap[String, Int] = {
    val res = textSet.generateWordIndexMap(removeTopN, maxWordsNum)
    if (res == null) null else res.asJava
  }

  def textSetIsDistributed(textSet: TextSet): Boolean = {
    textSet.isDistributed
  }

  def textSetIsLocal(textSet: TextSet): Boolean = {
    textSet.isLocal
  }

  def textSetGetTexts(textSet: LocalTextSet): JList[String] = {
    textSet.array.map(_.getText).toList.asJava
  }

  def textSetGetTexts(textSet: DistributedTextSet): JavaRDD[String] = {
    textSet.rdd.map(_.getText).toJavaRDD()
  }

  def textSetGetLabels(textSet: LocalTextSet): JList[Int] = {
    textSet.array.map(_.getLabel).toList.asJava
  }

  def textSetGetLabels(textSet: DistributedTextSet): JavaRDD[Int] = {
    textSet.rdd.map(_.getLabel).toJavaRDD()
  }

  def textSetGetPredicts(textSet: LocalTextSet): JList[JList[JTensor]] = {
    textSet.array.map{feature =>
      if (feature.contains(TextFeature.predict)) {
        activityToJTensors(feature[Activity](TextFeature.predict))
      }
      else {
        null
      }
    }.toList.asJava
  }

  def textSetGetPredicts(textSet: DistributedTextSet): JavaRDD[JList[JTensor]] = {
    textSet.rdd.map{feature =>
      if (feature.contains(TextFeature.predict)) {
        activityToJTensors(feature[Activity](TextFeature.predict))
      }
      else {
        null
      }
    }.toJavaRDD()
  }

  def textSetGetSamples(textSet: LocalTextSet): JList[Sample] = {
    textSet.array.map{feature =>
      if (feature.contains(TextFeature.sample)) {
        toPySample(feature.getSample.asInstanceOf[JSample[T]])
      }
      else {
        null
      }
    }.toList.asJava
  }

  def textSetGetSamples(textSet: DistributedTextSet): JavaRDD[Sample] = {
    textSet.rdd.map{feature =>
      if (feature.contains(TextFeature.sample)) {
        toPySample(feature.getSample.asInstanceOf[JSample[T]])
      }
      else {
        null
      }
    }.toJavaRDD()
  }

  def textSetRandomSplit(
      textSet: TextSet,
      weights: JList[Double]): JList[TextSet] = {
    textSet.randomSplit(weights.asScala.toArray).toList.asJava
  }

  def textSetTokenize(textSet: TextSet): TextSet = {
    textSet.tokenize()
  }

  def textSetNormalize(textSet: TextSet): TextSet = {
    textSet.normalize()
  }

  def textSetShapeSequence(
      textSet: TextSet,
      len: Int,
      mode: String): TextSet = {
    textSet.shapeSequence(len, toScalaTruncMode(mode))
  }

  def textSetWord2idx(
      textSet: TextSet,
      removeTopN: Int,
      maxWordsNum: Int): TextSet = {
    textSet.word2idx(removeTopN, maxWordsNum)
  }

  def textSetGenerateSample(textSet: TextSet): TextSet = {
    textSet.generateSample()
  }

  def textSetToDistributed(
      textSet: TextSet,
      sc: JavaSparkContext,
      partitionNum: Int = 4): DistributedTextSet = {
    textSet.toDistributed(sc.sc, partitionNum)
  }

  def textSetToLocal(textSet: TextSet): LocalTextSet = {
    textSet.toLocal()
  }

  def transformTextSet(
      transformer: Preprocessing[TextFeature, TextFeature],
      imageSet: TextSet): TextSet = {
    imageSet.transform(transformer)
  }

}
