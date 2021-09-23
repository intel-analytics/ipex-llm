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
 * Removes all dirty (non English alphabet) characters from tokens and converts words
 * to lower case. Need to tokenize first.
 * Input key: TextFeature.tokens
 * Output key: TextFeature.tokens
 *             In this case, original tokens will be replaced by normalized tokens.
 */
class Normalizer extends TextTransformer {

  override def transform(feature: TextFeature): TextFeature = {
    require(feature.contains(TextFeature.tokens), "TextFeature doesn't contain tokens yet, " +
      "please tokenize first")
    val tokens = feature[Array[String]](TextFeature.tokens)
    feature(TextFeature.tokens) = tokens.map(_.toLowerCase().replaceAll("[^a-z]", ""))
    feature
  }
}

object Normalizer {
  def apply(): Normalizer = {
    new Normalizer()
  }
}
