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

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.dataset.text.LabeledSentence
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class LabeledSentenceSpec extends FlatSpec with Matchers {
  "LabeledSentence with Float Array input and Array label" should "initialize well" in {
    val input = Array(0.0f, 1.0f, 2.0f, 3.0f)
    val label = Array(1.0f, 2.0f, 3.0f, 4.0f)

    val labeledSentence = new LabeledSentence[Float](input, label)

    labeledSentence.data() should be (input)

    labeledSentence.dataLength() should be (4)

    labeledSentence.label() should be (label)

    labeledSentence.labelLength() should be (4)
  }

  "LabeledSentence with Duble Array input and Array label" should "initialize well" in {
    val input = Array(0.0, 1.0, 2.0, 3.0)
    val label = Array(1.0, 2.0, 3.0, 4.0)

    val labeledSentence = new LabeledSentence[Double](input, label)

    labeledSentence.data() should be (input)

    labeledSentence.dataLength() should be (4)

    labeledSentence.label() should be (label)

    labeledSentence.labelLength() should be (4)
  }

  "LabeledSentence with Float Array input and Array label" should "copy well" in {
    val input = Array(0.0f, 1.0f, 2.0f, 3.0f)
    val label = Array(1.0f, 2.0f, 3.0f, 4.0f)

    val labeledSentence = new LabeledSentence[Float]()
    labeledSentence.copy(input, label)

    labeledSentence.data() should be (input)

    labeledSentence.dataLength() should be (4)

    labeledSentence.label() should be (label)

    labeledSentence.labelLength() should be (4)
  }

  "LabeledSentence with Double Array input and Array label" should "copy well" in {
    val input = Array(0.0, 1.0, 2.0, 3.0)
    val label = Array(1.0, 2.0, 3.0, 4.0)

    val labeledSentence = new LabeledSentence[Double]()
    labeledSentence.copy(input, label)

    labeledSentence.data() should be (input)

    labeledSentence.dataLength() should be (4)

    labeledSentence.label() should be (label)

    labeledSentence.labelLength() should be (4)
  }

  "LabeledSentence with Float Array input and Array label" should "clone well" in {
    val input = Array(0.0f, 1.0f, 2.0f, 3.0f)
    val label = Array(1.0f, 2.0f, 3.0f, 4.0f)

    val labeledSentence = new LabeledSentence[Float]()
    labeledSentence.copy(input, label)
    val otherLabeledSentence = labeledSentence.clone()

    labeledSentence.data() should be (otherLabeledSentence.data())

    labeledSentence.dataLength() should be (otherLabeledSentence.dataLength())

    labeledSentence.label() should be (otherLabeledSentence.label())

    labeledSentence.labelLength() should be (otherLabeledSentence.labelLength())
  }

  "LabeledSentence with Float Array input and Array label" should "be good in getData" +
    "and getLabel" in {
    val input = Array(0.0f, 1.0f, 2.0f, 3.0f)
    val label = Array(1.0f, 2.0f, 3.0f, 4.0f)

    val labeledSentence = new LabeledSentence[Float](input, label)

    for (i <- 0 until 4) {
      labeledSentence.getData(i) should be(input(i))
      labeledSentence.getLabel(i) should be(label(i))
    }
  }

  "LabeledSentence with Float Array input and Array label" should "be good in copyToData" +
    "and copyToLabel" in {
    val input = Array(0.0f, 1.0f, 2.0f, 3.0f)
    val label = Array(1.0f, 2.0f, 3.0f, 4.0f)

    val labeledSentence = new LabeledSentence[Float](input, label)

    val copyInput = new Array[Float](10)
    val copyLabel = new Array[Float](10)
    labeledSentence.copyToData(copyInput, 0)
    labeledSentence.copyToLabel(copyLabel, 0)

    for (i <- 0 until 4) {
      copyInput(i) should be (input(i))
      copyLabel(i) should be (label(i))
    }
  }

  "LabeledSentence with Float Array input and single label" should "initialize well" in {
    val input = Array(0.0f, 1.0f, 2.0f, 3.0f)
    val label = Array(1.0f)

    val labeledSentence = new LabeledSentence[Float](input, label)

    labeledSentence.data() should be (input)

    labeledSentence.dataLength() should be (4)

    labeledSentence.label() should be (label)

    labeledSentence.labelLength() should be (1)
  }

  "LabeledSentence with Float Array input and single label" should "copy well" in {
    val input = Array(0.0f, 1.0f, 2.0f, 3.0f)
    val label = Array(1.0f)

    val labeledSentence = new LabeledSentence[Float]()
    labeledSentence.copy(input, label)

    labeledSentence.data() should be (input)

    labeledSentence.dataLength() should be (4)

    labeledSentence.label() should be (label)

    labeledSentence.labelLength() should be (1)
  }

  "LabeledSentence with Float Array input and single label" should "clone well" in {
    val input = Array(0.0f, 1.0f, 2.0f, 3.0f)
    val label = Array(1.0f)

    val labeledSentence = new LabeledSentence[Float](input, label)
    val otherLabeledSentence = labeledSentence.clone()

    labeledSentence.data() should be (otherLabeledSentence.data())

    labeledSentence.dataLength() should be (otherLabeledSentence.dataLength())

    labeledSentence.label() should be (otherLabeledSentence.label())

    labeledSentence.labelLength() should be (otherLabeledSentence.labelLength())
  }
}
