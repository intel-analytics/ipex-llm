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

import org.scalatest.{FlatSpec, Matchers}

import scala.collection.immutable.HashSet

class TextFeatureSpec extends FlatSpec with Matchers {
  val text1 = "Hello my friend, please annotate my text"
  val text2 = "hello world, this is some sentence for my test"

  "TextFeature with label" should "work properly" in {
    val feature = TextFeature(text1, label = 0)
    require(feature.getText == text1)
    require(feature.hasLabel)
    require(feature.getLabel == 0)
    require(feature.keys() == HashSet("label", "text"))
    require(feature.getTokens == null)
    require(feature.getSample == null)
    require(feature.getPredict == null)
  }

  "TextFeature without label" should "work properly" in {
    val feature = TextFeature(text1)
    require(!feature.hasLabel)
    require(feature.getLabel == -1)
    require(feature.keys() == HashSet("text"))
  }
}
