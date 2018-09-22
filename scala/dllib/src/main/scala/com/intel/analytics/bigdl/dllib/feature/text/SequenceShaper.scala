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

package com.intel.analytics.zoo.feature.text

import com.intel.analytics.zoo.feature.text.TruncMode.TruncMode

/**
 * Shape the sequence of tokens to a fixed length.
 * If the original sequence is longer than the target length, it will be truncated from
 * the beginning or the end.
 * If the original sequence is shorter than the target length, it will be padded to the end.
 * Input key: TextFeature.tokens
 * Output key: TextFeature.tokens
 * The original token sequence will be replaced by the shaped sequence.
 *
 * @param len The target length.
 * @param truncMode Truncation mode. Either 'pre' or 'post'. Default is 'pre'.
 *                  If 'pre', the sequence will be truncated from the beginning.
 *                  If 'post', the sequence will be truncated from the end.
 * @param padElement String. The element to be padded to the sequence if the original length
 *                   is smaller than the target length.
 *                   Default is "##". Make sure that the padding element is meaningless in
 *                   your corpus.
 */
class SequenceShaper(
    val len: Int,
    val truncMode: TruncMode = TruncMode.pre,
    val padElement: String = "##") extends TextTransformer {

  require(len > 0, "len should be positive")

  override def transform(feature: TextFeature): TextFeature = {
    require(feature.contains(TextFeature.tokens), "TextFeature doesn't contain tokens, " +
      "please tokenize first")
    val tokens = feature[Array[String]](TextFeature.tokens)
    val shapedTokens = if (tokens.length > len) {
      truncMode match {
        case TruncMode.pre => tokens.slice(tokens.length - len, tokens.length)
        case TruncMode.post => tokens.slice(0, len)
        case _ => throw new IllegalArgumentException("Unknown truncation mode")
      }
    } else {
        tokens ++ Array.fill[String](len - tokens.length)(padElement)
    }
    feature(TextFeature.tokens) = shapedTokens
    feature
  }
}

object SequenceShaper {
  def apply(
      len: Int,
      truncMode: TruncMode = TruncMode.pre,
      padElement: String = "##"): SequenceShaper = {
    new SequenceShaper(len, truncMode, padElement)
  }
}

object TruncMode extends Enumeration {
  type TruncMode = Value
  val pre, post = Value
}
