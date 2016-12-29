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

package com.intel.analytics.bigdl.example.textclassification

import com.intel.analytics.bigdl.dataset.{MiniBatch, Transformer}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.Iterator

object SimpleTokenizer {
  /**
   * Simple tokenizer to split text into separated tokens.
   * @param text text to be split.
   * @param lower convert to lower case or not.
   * @return An array of separated tokens.
   */
  def toTokens(text: String, lower: Boolean = true): Array[String] = {
    text.replaceAll("[^a-zA-Z]", " ").toLowerCase().split("\\s+").filter(_.size > 2)
  }

  /**
   * Transform sample text into tokens and ignore those unknown tokens.
   * @param word2Meta Indicate the included words.
   */
  def toTokens(text: String, word2Meta: Map[String, WordMeta]): Array[Float] = {
    SimpleTokenizer.toTokens(text).map { word: String =>
      if (word2Meta.contains(word)) {
        Some(word2Meta(word).index.toFloat)
      } else {
        None
      }
    }.flatten
  }

  /**
   * Shape the token sequence to the specified length.
   * The sequence would be either padded or truncated.
   * @param sequenceLen the desired seq length
   * @param trunc truncated from pre or post.
   */
  def shaping(tokens: Array[Float], sequenceLen: Int, trunc: String = "pre")
  : Array[Float] = {
    val paddedTokens = if (tokens.length > sequenceLen) {
      if ("pre" == trunc) {
        tokens.slice(tokens.length - sequenceLen, tokens.length)
      } else {
        tokens.slice(0, sequenceLen)
      }
    } else {
      tokens ++ Array.fill[Float](sequenceLen - tokens.length)(0)
    }
    paddedTokens
  }

  /**
   * Transform word to pre-trained vector.
   * @param embeddingSize size of the pre-trained vector
   * @param word2Vec pre-trained word2Vec
   */
  def vectorization(tokens: Array[Float], embeddingSize: Int, word2Vec: Map[Float, Array[Float]])
  : Array[Array[Float]] = {
    tokens.map { word =>
      if (word2Vec.contains(word)) {
        word2Vec(word)
      } else {
        // Treat it as zeros if cannot be found from pre-trained word2Vec
        Array.fill[Float](embeddingSize)(0)
      }
    }
  }
}

object Batching {
  def apply(batchSize: Int, sampleShape: Array[Int]): Batching =
    new Batching(batchSize, sampleShape)
}

/**
 * Batching samples into mini-batch
 * @param batchSize The desired mini-batch size.
 * @param sampleShape Shape of the training sample
 */
class Batching(batchSize: Int, sampleShape: Array[Int])
  extends Transformer[(Array[Array[Float]], Float), MiniBatch[Float]] {

  val log: Logger = LoggerFactory.getLogger(this.getClass)

  override def apply(prev: Iterator[(Array[Array[Float]], Float)]): Iterator[MiniBatch[Float]] = {
    new Iterator[MiniBatch[Float]] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null
      private var labelData: Array[Float] = null
      private val sampleSize = sampleShape.product

      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[Float] = {
        var i = 0
        while (i < batchSize && prev.hasNext) {
          val sample = prev.next()
          if (featureData == null) {
            featureData = new Array[Float](batchSize * sampleSize)
            labelData = new Array[Float](batchSize)
          }
          Array.copy(sample._1.flatten, 0,
            featureData, i * sampleSize, sampleSize)
          labelData(i) = sample._2
          i += 1
        }
        featureTensor.set(Storage[Float](featureData),
          storageOffset = 1, sizes = Array(i) ++ sampleShape)
        labelTensor.set(Storage[Float](labelData),
          storageOffset = 1, sizes = Array(i))
        MiniBatch(featureTensor.transpose(2, 3), labelTensor)
      }
    }
  }
}

/**
 * @param count frequency of the word.
 * @param index index of the word which ranked by the frequency from high to low.
 */
case class WordMeta(count: Int, index: Int)
