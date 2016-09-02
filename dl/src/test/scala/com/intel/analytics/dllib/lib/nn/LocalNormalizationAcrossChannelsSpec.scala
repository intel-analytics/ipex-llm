package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.{torch, Tensor}
import org.scalatest.{Matchers, FlatSpec}

class LocalNormalizationAcrossChannelsSpec extends FlatSpec with Matchers {
  private def referenceLRNForwardAcrossChannels
  (input : Tensor[Double], alpha : Double, beta : Double, size : Int) : Tensor[Double] = {
    val output = torch.Tensor[Double]()
    output.resizeAs(input)
    val batch = input.size(1)
    val channel = input.size(2)
    val height = input.size(3)
    val width = input.size(4)

    for(n <- 0 until batch) {
      for(c <- 0 until channel) {
        for(h <- 0 until height) {
          for(w <- 0 until width) {
            var cStart = c - (size - 1) / 2
            val cEnd = math.min(cStart + size, channel)
            cStart = math.max(cStart, 0)
            var scale = 1.0
            for(i <- cStart until cEnd) {
              val value = input.valueAt(n + 1, i + 1, h + 1, w + 1)
              scale += value * value * alpha / size
            }
            output.setValue(n + 1, c + 1, h + 1, w + 1, input.valueAt(n + 1, c + 1, h + 1, w + 1) * math.pow(scale, -beta))
          }
        }
      }
    }

    output
  }

  private def referenceLRNForwardAcrossChannels
  (input : Tensor[Float], alpha : Float, beta : Float, size : Int) : Tensor[Float] = {
    val output = torch.Tensor[Float]()
    output.resizeAs(input)
    val batch = input.size(1)
    val channel = input.size(2)
    val height = input.size(3)
    val width = input.size(4)

    for(n <- 0 until batch) {
      for(c <- 0 until channel) {
        for(h <- 0 until height) {
          for(w <- 0 until width) {
            var cStart = c - (size - 1) / 2
            val cEnd = math.min(cStart + size, channel)
            cStart = math.max(cStart, 0)
            var scale = 1.0f
            for(i <- cStart until cEnd) {
              val value = input.valueAt(n + 1, i + 1, h + 1, w + 1)
              scale += value * value * alpha / size
            }
            output.setValue(n + 1, c + 1, h + 1, w + 1, input.valueAt(n + 1, c + 1, h + 1, w + 1) * math.pow(scale, -beta).toFloat)
          }
        }
      }
    }

    output
  }

  "LocalNormalizationAcrossChannels Foward Double" should "be correct" in {
    val layer = new LocalNormalizationAcrossChannels[Double](5, 0.0001, 0.75, 1.0)
    val input = torch.Tensor[Double](2, 7, 3, 3)
    input.rand()
    val outputRef = referenceLRNForwardAcrossChannels(input, 0.0001, 0.75, 5)
    layer.forward(input)
    val output = layer.forward(input)

    var diff = 0.0
    output.map(outputRef, (a, b) => {diff += math.abs(a - b) ; a})
    diff should be(0.0)
  }

  "LocalNormalizationAcrossChannels BackWard Double" should "be correct" in {
    val layer = new LocalNormalizationAcrossChannels[Double](5, 0.0001, 0.75, 1.0)
    val input = torch.Tensor[Double](2, 7, 3, 3)
    input.rand()
    val checker = new GradientChecker(1e-2, 1e-2)
    checker.checkLayer(layer, input) should be(true)
  }

  "LocalNormalizationAcrossChannels BackWard Float" should "be correct" in {
    val layer = new LocalNormalizationAcrossChannels[Float](5, 0.0001, 0.75, 1.0)
    val input = torch.Tensor[Float](2, 7, 3, 3)
    input.rand()
    val checker = new GradientChecker(1e-2, 1e-2)
    checker.checkLayer(layer, input) should be(true)
  }

  "LocalNormalizationAcrossChannels with Large Region BackWard Double" should "be correct" in {
    val layer = new LocalNormalizationAcrossChannels[Double](15, 0.0001, 0.75, 1.0)
    val input = torch.Tensor[Double](2, 7, 3, 3)
    input.rand()
    val checker = new GradientChecker(1e-2, 1e-2)
    checker.checkLayer(layer, input) should be(true)
  }

  "LocalNormalizationAcrossChannels with Large Region BackWard Float" should "be correct" in {
    val layer = new LocalNormalizationAcrossChannels[Float](15, 0.0001, 0.75, 1.0)
    val input = torch.Tensor[Float](2, 7, 3, 3)
    input.rand()
    val checker = new GradientChecker(1e-2, 1e-2)
    checker.checkLayer(layer, input) should be(true)
  }

  "LocalNormalizationAcrossChannels with Large Region Foward Double" should "be correct" in {
    val layer = new LocalNormalizationAcrossChannels[Double](15, 0.0001, 0.75, 1.0)
    val input = torch.Tensor[Double](2, 7, 3, 3)
    input.rand()
    val outputRef = referenceLRNForwardAcrossChannels(input, 0.0001, 0.75, 15)
    val output = layer.forward(input)

    var diff = 0.0
    output.map(outputRef, (a, b) => {diff += math.abs(a - b) ; a})
    diff should be(0.0)
  }

  "LocalNormalizationAcrossChannels Foward Float" should "be correct" in {
    val layer = new LocalNormalizationAcrossChannels[Float](5, 0.0001f, 0.75f, 1.0f)
    val input = torch.Tensor[Float](2, 7, 3, 3)
    input.rand()
    val outputRef = referenceLRNForwardAcrossChannels(input, 0.0001f, 0.75f, 5)
    val output = layer.forward(input)

    var diff = 0.0f
    output.map(outputRef, (a, b) => {diff += math.abs(a - b) ; a})
    diff should be(0.0f)
  }

  "LocalNormalizationAcrossChannels with Large Region Foward Float" should "be correct" in {
    val layer = new LocalNormalizationAcrossChannels[Float](15, 0.0001f, 0.75f, 1.0f)
    val input = torch.Tensor[Float](2, 7, 3, 3)
    input.rand()
    val outputRef = referenceLRNForwardAcrossChannels(input, 0.0001f, 0.75f, 15)
    val output = layer.forward(input)

    var diff = 0.0f
    output.map(outputRef, (a, b) => {diff += math.abs(a - b) ; a})
    diff should be(0.0f)
  }
}
