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

/**
 * Given a wordIndex map, transform tokens to corresponding indices.
 * Those words not in the map will be aborted.
 * Need to tokenize first.
 * Input key: TextFeature.tokens
 * Output key: TextFeature.indexedTokens
 *
 * @param map Map of each word (String) and its index (integer).
 */
class WordIndexer(val map: Map[String, Int]) extends TextTransformer {

  require(map != null, "map for WordIndexer can't be null")

  override def transform(feature: TextFeature): TextFeature = {
    require(feature.contains(TextFeature.tokens), "TextFeature doesn't contain tokens yet, " +
      "please tokenize first")
    val tokens = feature.getTokens
    val indices = tokens.flatMap(word => {
      if (map.contains(word)) {
        Some(map(word).toFloat)
      }
      else {
        None
      }
    })
    feature(TextFeature.indexedTokens) = indices
    feature
  }
}

object WordIndexer {
  def apply(wordIndex: Map[String, Int]): WordIndexer = {
    new WordIndexer(wordIndex)
  }
}
