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

/**
 * Given a wordIndex map, transform tokens to corresponding indices.
 * Input key: TextFeature.tokens
 * Output key: TextFeature.indexedTokens
 *
 * @param map Map of each word (String) and its index (integer).
 *            It is recommended that the map contains all the words in your corpus.
 * @param replaceElement Integer. The element to fill if the word is not in
 *                       the given map. Default is 0 with the convention that
 *                       0 is reserved for unknown words.
 */
class WordIndexer(
   val map: Map[String, Int],
   val replaceElement: Int = 0) extends TextTransformer {

  override def transform(feature: TextFeature): TextFeature = {
    require(feature.contains(TextFeature.tokens), "TextFeature doesn't contain tokens yet, " +
      "please tokenize first")
    val tokens = feature[Array[String]](TextFeature.tokens)
    val indices = tokens.map(word => {
      if (map.contains(word)) {
        map(word).toFloat
      }
      else {
        replaceElement.toFloat
      }
    })
    feature(TextFeature.indexedTokens) = indices
    feature
  }
}

object WordIndexer {
  def apply(
     wordIndex: Map[String, Int],
     replaceElement: Int = 0): WordIndexer = {
    new WordIndexer(wordIndex)
  }
}
