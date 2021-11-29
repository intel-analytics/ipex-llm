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
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Transform indexedTokens and label (if any) of a TextFeature to a BigDL Sample.
 * Need to word2idx first.
 * Input key: TextFeature.indexedTokens and TextFeature.label (if any)
 * Output key: TextFeature.sample
 */
class TextFeatureToSample extends TextTransformer {

  override def transform(feature: TextFeature): TextFeature = {
    require(feature.contains(TextFeature.indexedTokens), "TextFeature doesn't contain indexTokens" +
      " yet. Please use WordIndexer to transform tokens to indexedTokens first")
    val indices = feature.getIndices
    val input = Tensor[Float](data = indices, shape = Array(indices.length))
    val sample = if (feature.hasLabel) {
      Sample[Float](input, feature.getLabel.toFloat)
    }
    else {
      Sample[Float](input)
    }
    feature(TextFeature.sample) = sample
    feature
  }
}

object TextFeatureToSample {
  def apply(): TextFeatureToSample = {
    new TextFeatureToSample()
  }
}
