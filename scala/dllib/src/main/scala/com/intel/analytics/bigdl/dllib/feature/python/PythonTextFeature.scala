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

import com.intel.analytics.bigdl.python.api.{PythonBigDL, Sample}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.feature.text.TruncMode.TruncMode
import com.intel.analytics.zoo.feature.text._

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
    if (tokens != null ) {
      tokens.toList.asJava
    }
    else {
      null
    }
  }

  def textFeatureGetSample(feature: TextFeature): Sample = {
    val sample = feature.getSample[T]
    if (sample != null) {
      toPySample(sample)
    }
    else {
      null
    }
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
}
