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

package com.intel.analytics.bigdl.example.utils

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

/**
 * @param count frequency of the word.
 * @param index index of the word which ranked by the frequency from high to low.
 */
case class WordMeta(count: Int, index: Int)
