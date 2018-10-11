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

import org.scalatest.{FlatSpec, Matchers}

class TextFeatureToSampleSpec extends FlatSpec with Matchers {

  private def genFeature(): TextFeature = {
    val text = "hello my friend, please annotate my text"
    val feature = TextFeature(text)
    feature(TextFeature.indexedTokens) = Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 6.0f)
    feature
  }

  "TextFeatureToSample with label" should "work properly" in {
    val toSample = TextFeatureToSample()
    val transformed = toSample.transform(genFeature().setLabel(1))
    val sample = transformed.getSample
    require(sample.getData().sameElements(Array(1.0f, 2.0f,
      3.0f, 4.0f, 5.0f, 2.0f, 6.0f, 1.0f)))
  }

  "TextFeatureToSample without label" should "work properly" in {
    val toSample = TextFeatureToSample()
    val transformed = toSample.transform(genFeature())
    val sample = transformed.getSample
    require(sample.getData().sameElements(Array(1.0f, 2.0f,
      3.0f, 4.0f, 5.0f, 2.0f, 6.0f)))
  }
}
