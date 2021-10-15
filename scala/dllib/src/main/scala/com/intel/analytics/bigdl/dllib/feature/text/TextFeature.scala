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

package com.intel.analytics.bigdl.dllib.feature.text

import com.intel.analytics.bigdl.dllib.feature.dataset.Sample
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import org.apache.log4j.Logger

import scala.collection.{Set, mutable}
import scala.reflect.ClassTag

/**
 * Each TextFeature keeps information of a single text record.
 * It can include various status (if any) of a text,
 * e.g. original text content, uri, category label, tokens, index representation
 * of tokens, BigDL Sample representation, prediction result and so on.
 * It uses a HashMap to store all these data.
 * Each key is a string that can be used to identify the corresponding value.
 */
class TextFeature extends Serializable {
  import TextFeature.logger

  private def this(text: String, label: Option[Int], uri: String) {
    this
    if (text != null) {
      state(TextFeature.text) = text
    }
    if (label.nonEmpty) {
      state(TextFeature.label) = label.get
    }
    if (uri != null) {
      state(TextFeature.uri) = uri
    }
  }

  private val state = new mutable.HashMap[String, Any]()

  def contains(key: String): Boolean = state.contains(key)

  def apply[T](key: String): T = {
    if (contains(key)) {
      state(key).asInstanceOf[T]
    } else {
      logger.warn(s"TextFeature doesn't contain $key")
      null.asInstanceOf[T]
    }
  }

  def update(key: String, value: Any): Unit = state(key) = value

  def keys(): Set[String] = state.keySet

  /**
   * Whether the TextFeature contains label.
   */
  def hasLabel: Boolean = state.contains(TextFeature.label)

  /**
   * Set the label for the TextFeature.
   * @param label Integer.
   * @return The TextFeature with label.
   */
  def setLabel(label: Int): this.type = {
    if (hasLabel) {
      logger.warn(s"Label exists, overwriting the original label $label")
    }
    state(TextFeature.label) = label
    this
  }

  /**
   * Get the label of the TextFeature.
   * If no label is stored, -1 will be returned.
   */
  def getLabel: Int = {
    if (hasLabel) {
      apply[Int](TextFeature.label)
    }
    else {
      logger.warn("No label is stored in the TextFeature")
      -1
    }
  }

  /**
   * Get the text content of the TextFeature.
   * Return null if it doesn't exist.
   */
  def getText: String = apply[String](TextFeature.text)

  /**
   * Get the identifier of the TextFeature.
   * Return null if it doesn't exist.
   */
  def getURI: String = apply[String](TextFeature.uri)

  /**
   * Get the tokens of the TextFeature.
   * If text hasn't been segmented, null will be returned.
   */
  def getTokens: Array[String] = apply[Array[String]](TextFeature.tokens)

  /**
   * Get the token indices of the TextFeature.
   * If text hasn't been segmented or transformed from word to index, null will be returned.
   */
  def getIndices: Array[Float] = apply[Array[Float]](TextFeature.indexedTokens)

  /**
   * Get the Sample representation of the TextFeature.
   * If the TextFeature hasn't been transformed to Sample, null will be returned.
   */
  def getSample: Sample[Float] = apply[Sample[Float]](TextFeature.sample)

  /**
   * Get the prediction probability distribution of the TextFeature.
   * If the TextFeature hasn't been predicted by a model, null will be returned.
   */
  def getPredict[T: ClassTag]: Tensor[T] = apply[Tensor[T]](TextFeature.predict)
}

object TextFeature {
  /**
   * Key for the identifier of the TextFeature.
   * It can be the id of the text in your corpus or the uri of the file.
   * Value should be a String.
   */
  val uri = "uri"
  /**
   * Key for the original text content.
   * Value should be a String.
   */
  val text = "text"
  /**
   * Key for the label for the original text content.
   * Value should be an integer.
   */
  val label = "label"
  /**
   * Key for the tokens after doing tokenization (or other token-based transformation such as
   * normalization) on the original text.
   * Value should be an array of String.
   */
  val tokens = "tokens"
  /**
   * Key for the indices corresponding to the tokens after performing word2idx.
   * Value should be an array of Float.
   */
  val indexedTokens = "indexedTokens"
  /**
   * Key for the sample (feature and label if any).
   * Value should be a BigDL Sample[Float].
   */
  val sample = "sample"
  /**
   * Key for the text prediction result.
   * Value should be a BigDL Tensor.
   */
  val predict = "predict"

  val logger: Logger = Logger.getLogger(getClass)

  /**
   * Create a TextFeature without label.
   */
  def apply(text: String, uri: String = null): TextFeature = {
    new TextFeature(text, None, uri)
  }

  /**
   * Create a TextFeature with label.
   * It is recommended that label starts from 0.
   */
  def apply(text: String, label: Int): TextFeature = {
    new TextFeature(text, Some(label), null)
  }

  /**
   * Create a TextFeature with label and uri.
   * It is recommended that label starts from 0.
   */
  def apply(text: String, label: Int, uri: String): TextFeature = {
    new TextFeature(text, Some(label), uri)
  }
}
