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

import com.intel.analytics.bigdl.dllib.feature.text.TruncMode.TruncMode

/**
 * Shape the sequence of indices to a fixed length.
 * If the original sequence is longer than the target length, it will be truncated from
 * the beginning or the end.
 * If the original sequence is shorter than the target length, it will be padded to the end.
 * Need to word2idx first.
 * Input key: TextFeature.indexedTokens
 * Output key: TextFeature.indexedTokens
 * The original indices sequence will be replaced by the shaped sequence.
 *
 * @param len Positive integer. The target length.
 * @param truncMode Truncation mode. Either TruncMode.pre or TruncMode.post.
 *                  If TruncMode.pre, the sequence will be truncated from the beginning.
 *                  If TruncMode.post, the sequence will be truncated from the end.
 *                  Default is TruncMode.pre.
 * @param padElement Integer. The index element to be padded to the sequence if the original
 *                   length is smaller than the target length.
 *                   Default is 0 with the convention that we reserve index 0 for unknown words.
 */
class SequenceShaper(
    val len: Int,
    val truncMode: TruncMode = TruncMode.pre,
    val padElement: Int = 0) extends TextTransformer {

  require(len > 0, "len should be positive")

  override def transform(feature: TextFeature): TextFeature = {
    require(feature.contains(TextFeature.indexedTokens), "TextFeature doesn't contain " +
      "indexedTokens, please transform from word to index first")
    val indices = feature.getIndices
    val shapedIndices = if (indices.length > len) {
      truncMode match {
        case TruncMode.pre => indices.slice(indices.length - len, indices.length)
        case TruncMode.post => indices.slice(0, len)
        case _ => throw new IllegalArgumentException("Unknown truncation mode")
      }
    } else {
        indices ++ Array.fill[Float](len - indices.length)(padElement)
    }
    feature(TextFeature.indexedTokens) = shapedIndices
    feature
  }
}

object SequenceShaper {
  def apply(
      len: Int,
      truncMode: TruncMode = TruncMode.pre,
      padElement: Int = 0): SequenceShaper = {
    new SequenceShaper(len, truncMode, padElement)
  }
}

object TruncMode extends Enumeration {
  type TruncMode = Value
  val pre, post = Value
}
