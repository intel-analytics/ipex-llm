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
 * Transform text to array of string tokens.
 * Input key: TextFeature.text
 * Output key: TextFeature.tokens
 */
class Tokenizer extends TextTransformer {

  override def transform(feature: TextFeature): TextFeature = {
    val text = feature[String](TextFeature.text)
    feature(TextFeature.tokens) = text.split("\\s+")
    feature
  }
}

object Tokenizer {
  def apply(): Tokenizer = {
    new Tokenizer()
  }
}
