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

package com.intel.analytics.bigdl.dataset.image

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

import scala.collection.Iterator
import scala.util.Random

object ColorJitter {
  def apply(): ColorJitter = {
    new ColorJitter()
  }
}

/**
 * Process an image with brightness, contrast, saturation in a random order
 */
class ColorJitter extends Transformer[LabeledBGRImage, LabeledBGRImage] {
  // Todo: make the bcs parameter configurable
  private val bcsParameters = Map("brightness" -> 0.4f, "contrast" -> 0.4f, "saturation" -> 0.4f)

  private def grayScale(dst: Array[Float], img: Array[Float]): Array[Float] = {
    var i = 0
    while (i < img.length) {
      dst(i) = img(i)*0.299f + img(i + 1)*0.587f + img(i + 2)*0.114f
      dst(i + 1) = dst(i)
      dst(i + 2) = dst(i)
      i += 3
    }
    dst
  }

  private def blend(img1: Array[Float], img2: Array[Float], alpha: Float): Unit =
    (img1 zip img2) map {case (a, b) => a + (1-alpha)*b }

  private def saturation(variance: Float)(input: Array[Float]): Array[Float] = {
    val gs = new Array[Float](input.length)
    grayScale(gs, input)
    val alpha = 1.0f + RNG.uniform(-variance, variance).toFloat
    blend(input, gs, alpha)
    input
  }

  private def brightness(variance: Float)(input: Array[Float]): Array[Float] = {
    val gs = new Array[Float](input.length)
    val alpha = 1.0f + RNG.uniform(-variance, variance).toFloat
    blend(input, gs, alpha)
    input
  }

  private def contrast(variance: Float)(input: Array[Float]): Array[Float] = {
    val gs = new Array[Float](input.length)
    grayScale(gs, input)
    val mean = gs.sum / gs.length
    gs.foreach( _ => mean)
    val alpha = 1.0f + RNG.uniform(-variance, variance).toFloat
    blend(input, gs, alpha)
    input
  }

  private val ts = Map(
    "brightness" -> {brightness(bcsParameters.get("brightness").get)(_)},
    "contrast" -> {contrast(bcsParameters.get("contrast").get)(_)},
    "saturation" -> {saturation(bcsParameters.get("saturation").get)(_)}
  )

  private def randomOrder(input: Array[Float]): Unit = {
    val randOrder = Random.shuffle(List("brightness", "contrast", "saturation"))
    randOrder.map( x => ts(x))
  }

  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[LabeledBGRImage] = {
    prev.map(img => {
      val content = img.content
      require(content.length % 3 == 0)
      randomOrder(content)
      img
    })
  }
}
