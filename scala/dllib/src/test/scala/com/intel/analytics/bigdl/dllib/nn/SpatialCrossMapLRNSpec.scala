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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class SpatialCrossMapLRNSpec extends FlatSpec with Matchers {
  private def referenceLRNForwardAcrossChannels
  (input: Tensor[Double], alpha: Double, beta: Double, size: Int): Tensor[Double] = {
    val output = Tensor[Double]()
    output.resizeAs(input)
    val batch = input.size(1)
    val channel = input.size(2)
    val height = input.size(3)
    val width = input.size(4)

    for (n <- 0 until batch) {
      for (c <- 0 until channel) {
        for (h <- 0 until height) {
          for (w <- 0 until width) {
            var cStart = c - (size - 1) / 2
            val cEnd = math.min(cStart + size, channel)
            cStart = math.max(cStart, 0)
            var scale = 1.0
            for (i <- cStart until cEnd) {
              val value = input.valueAt(n + 1, i + 1, h + 1, w + 1)
              scale += value * value * alpha / size
            }
            output.setValue(n + 1, c + 1, h + 1, w + 1,
              input.valueAt(n + 1, c + 1, h + 1, w + 1) * math.pow(scale, -beta))
          }
        }
      }
    }

    output
  }

  private def referenceLRNForwardAcrossChannels
  (input: Tensor[Float], alpha: Float, beta: Float, size: Int): Tensor[Float] = {
    val output = Tensor[Float]()
    output.resizeAs(input)
    val batch = input.size(1)
    val channel = input.size(2)
    val height = input.size(3)
    val width = input.size(4)

    for (n <- 0 until batch) {
      for (c <- 0 until channel) {
        for (h <- 0 until height) {
          for (w <- 0 until width) {
            var cStart = c - (size - 1) / 2
            val cEnd = math.min(cStart + size, channel)
            cStart = math.max(cStart, 0)
            var scale = 1.0f
            for (i <- cStart until cEnd) {
              val value = input.valueAt(n + 1, i + 1, h + 1, w + 1)
              scale += value * value * alpha / size
            }
            output.setValue(n + 1, c + 1, h + 1, w + 1,
              input.valueAt(n + 1, c + 1, h + 1, w + 1) * math.pow(scale, -beta).toFloat)
          }
        }
      }
    }

    output
  }

  "LocalNormalizationAcrossChannels Forward Double" should "be correct" in {
    val layer = new SpatialCrossMapLRN[Double](5, 0.0001, 0.75, 1.0)
    val input = Tensor[Double](2, 7, 3, 3)
    input.rand()
    val outputRef = referenceLRNForwardAcrossChannels(input, 0.0001, 0.75, 5)
    layer.forward(input)
    val output = layer.forward(input)

    output should be(outputRef)
  }

  "LocalNormalizationAcrossChannels Backward Double" should "be correct" in {
    val layer = new SpatialCrossMapLRN[Double](5, 0.0001, 0.75, 1.0)
    val input = Tensor[Double](2, 7, 3, 3)
    input.rand()
    val checker = new GradientChecker(1e-2, 1e-2)
    checker.checkLayer(layer, input) should be(true)
  }

  "LocalNormalizationAcrossChannels Backward Float" should "be correct" in {
    val layer = new SpatialCrossMapLRN[Float](5, 0.0001, 0.75, 1.0)
    val input = Tensor[Float](2, 7, 3, 3)
    input.rand()
    val checker = new GradientChecker(1e-2, 1e-2)
    checker.checkLayer[Float](layer, input) should be(true)
  }

  "LocalNormalizationAcrossChannels with Large Region Backward Double" should "be correct" in {
    val layer = new SpatialCrossMapLRN[Double](15, 0.0001, 0.75, 1.0)
    val input = Tensor[Double](2, 7, 3, 3)
    input.rand()
    val checker = new GradientChecker(1e-2, 1e-2)
    checker.checkLayer(layer, input) should be(true)
  }

  "LocalNormalizationAcrossChannels with Large Region Backward Float" should "be correct" in {
    val layer = new SpatialCrossMapLRN[Float](15, 0.0001, 0.75, 1.0)
    val input = Tensor[Float](2, 7, 3, 3)
    input.rand()
    val checker = new GradientChecker(1e-2, 1e-2)
    checker.checkLayer(layer, input) should be(true)
  }

  "LocalNormalizationAcrossChannels with Large Region Forward Double" should "be correct" in {
    val layer = new SpatialCrossMapLRN[Double](15, 0.0001, 0.75, 1.0)
    val input = Tensor[Double](2, 7, 3, 3)
    input.rand()
    val outputRef = referenceLRNForwardAcrossChannels(input, 0.0001, 0.75, 15)
    val output = layer.forward(input)

    output should be(outputRef)
  }

  "LocalNormalizationAcrossChannels Forward Float" should "be correct" in {
    val layer = new SpatialCrossMapLRN[Float](5, 0.0001f, 0.75f, 1.0f)
    val input = Tensor[Float](2, 7, 3, 3)
    input.rand()
    val outputRef = referenceLRNForwardAcrossChannels(input, 0.0001f, 0.75f, 5)
    val output = layer.forward(input)

    output should be(outputRef)
  }

  "LocalNormalizationAcrossChannels with Large Region Forward Float" should "be correct" in {
    val layer = new SpatialCrossMapLRN[Float](15, 0.0001f, 0.75f, 1.0f)
    val input = Tensor[Float](2, 7, 3, 3)
    input.rand()
    val outputRef = referenceLRNForwardAcrossChannels(input, 0.0001f, 0.75f, 15)
    val output = layer.forward(input)

    output should be(outputRef)
  }
}

class SpatialCrossMapLRNSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val spatialCrossMapLRN = SpatialCrossMapLRN[Float](5, 0.01, 0.75, 1.0).
      setName("spatialCrossMapLRN")
    val input = Tensor[Float](2, 2, 2, 2).apply1( e => Random.nextFloat())
    runSerializationTest(spatialCrossMapLRN, input)
  }
}
