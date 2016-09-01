package com.intel.webscaleml.nn.nn

import java.util

import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric
import com.intel.webscaleml.nn.tensor.{Tensor, torch}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect._

class LocalNormalizationAcrossChannels[@specialized(Float, Double) T: ClassTag]
  (val size : Int = 5, val alpha : Double = 1.0, val beta : Double = 0.75, val k : Double = 1.0)(
    implicit ev: TensorNumeric[T]) extends Module[T] {

  private val scale = torch.Tensor[T]()
  private val paddedSquare = torch.Tensor[T]()
  private val paddedRatio = torch.Tensor[T]()
  private val accumRatio = torch.Tensor[T]()
  private val accumRatioTimeInput = torch.Tensor[T]()
  private var results : Array[Future[Unit]] = null

  require(size % 2 == 1, "LRN only supports odd values for size")
  val prePad = (size - 1) / 2

  override def equals(obj : Any) : Boolean = {
    if(!super.equals(obj)) {
      return false
    }

    if(!obj.isInstanceOf[LocalNormalizationAcrossChannels[T]])
      return false
    val other = obj.asInstanceOf[LocalNormalizationAcrossChannels[T]]
    if(this.eq(other))
      return true

    size == other.size &&
      alpha == other.alpha && beta == other.beta && k == other.k
  }

  override def toString() : String = {
    s"nn.LocalResponseNormalizationAcrossChannels($size, $alpha, $beta, $k)"
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "Input must have 4 dimensions, corresponding to (batch, channels, height, width)")
    require(input.isContiguous(), "Input is not contiguous")

    output.resizeAs(input)
    scale.resizeAs(input)

    val batchNum = input.size(1)
    val channel = input.size(2)
    val height = input.size(3)
    val width = input.size(4)
    paddedSquare.resize(batchNum, channel + size - 1, height, width)

    if(results == null || results.length != batchNum) {
      results = new Array[Future[Unit]](batchNum)
    }

    if (classTag[T] == classTag[Double]) {
      LocalNormalizationAcrossChannels.LRNForwardDouble(
        input.asInstanceOf[Tensor[Double]], output.asInstanceOf[Tensor[Double]],
        paddedSquare.asInstanceOf[Tensor[Double]], scale.asInstanceOf[Tensor[Double]], prePad, alpha,
        size, beta, k, results
      )
    } else if(classTag[T] == classTag[Float]) {
      LocalNormalizationAcrossChannels.LRNForwardFloat(
        input.asInstanceOf[Tensor[Float]], output.asInstanceOf[Tensor[Float]],
        paddedSquare.asInstanceOf[Tensor[Float]], scale.asInstanceOf[Tensor[Float]], prePad, alpha.toFloat,
        size, beta.toFloat, k.toFloat, results
      )
    } else {
      ???
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "Input must have 4 dimensions, corresponding to (batch, channels, height, width)")
    require(gradOutput.isContiguous(), "gradOutput is not contiguous")

    val batchNum = input.size(1)
    val channel = input.size(2)
    val height = input.size(3)
    val width = input.size(4)

    paddedRatio.resize(batchNum, channel + size - 1, height, width)
    accumRatio.resize(batchNum, 1, height, width)
    gradInput.resizeAs(input)
    accumRatioTimeInput.resize(batchNum, 1, height, width)

    if(results == null || results.length != batchNum) {
      results = new Array[Future[Unit]](batchNum)
    }

    if (classTag[T] == classTag[Double]) {
      LocalNormalizationAcrossChannels.LRNBackwardDouble(
        input.asInstanceOf[Tensor[Double]], output.asInstanceOf[Tensor[Double]], gradOutput.asInstanceOf[Tensor[Double]],
        gradInput.asInstanceOf[Tensor[Double]], paddedRatio.asInstanceOf[Tensor[Double]], scale.asInstanceOf[Tensor[Double]],
        accumRatio.asInstanceOf[Tensor[Double]], accumRatioTimeInput.asInstanceOf[Tensor[Double]], size, alpha,
        beta, results
      )
    } else if(classTag[T] == classTag[Float]) {
      LocalNormalizationAcrossChannels.LRNBackwardFloat(
        input.asInstanceOf[Tensor[Float]], output.asInstanceOf[Tensor[Float]], gradOutput.asInstanceOf[Tensor[Float]],
        gradInput.asInstanceOf[Tensor[Float]], paddedRatio.asInstanceOf[Tensor[Float]], scale.asInstanceOf[Tensor[Float]],
        accumRatio.asInstanceOf[Tensor[Float]], accumRatioTimeInput.asInstanceOf[Tensor[Float]], size, alpha.toFloat,
        beta.toFloat, results
      )
    } else {
      ???
    }

    this.gradInput
  }
}

object LocalNormalizationAcrossChannels {
  private def LRNBackwardDouble(
    input : Tensor[Double], output : Tensor[Double], gradOutput : Tensor[Double],
    gradInput : Tensor[Double], paddedRatio : Tensor[Double], scale : Tensor[Double],
    accumRatio : Tensor[Double], accumRatioTimeInput : Tensor[Double],
    size : Int, alpha : Double, beta : Double, results : Array[Future[Unit]]) : Unit = {

    val batchNum = input.size(1)
    val channel = input.size(2)
    val height = input.size(3)
    val width = input.size(4)

    val paddedRatioData = paddedRatio.storage().array()
    val gradInputData = gradInput.storage().array()
    val gradOutputData = gradOutput.storage().array()
    val outputData = output.storage().array()
    val scaleData = scale.storage().array()
    val accumRatioData = accumRatio.storage().array()
    val accumRationTimeInputData = accumRatioTimeInput.storage().array()
    val inputData = input.storage().array()
    val ratioValue = 2.0 * alpha * beta / size
    val inversePrePad = size - (size + 1) / 2
    var i = 0
    while(i < batchNum) {
      val b = i + 1
      results(i) = Future {
        val gradInputOffset = gradInput.select(1, b).storageOffset() - 1
        val gradOutputOffset = gradOutput.select(1, b).storageOffset() - 1
        val scaleOffset = scale.select(1, b).storageOffset() - 1

        var j = 0
        while(j < channel * height * width) {
          gradInputData(gradInputOffset + j) = math.pow(scaleData(scaleOffset + j), -beta)
          gradInputData(gradInputOffset + j) *= gradOutputData(gradOutputOffset + j)
          j += 1
        }

        val paddedRatioOffset = paddedRatio.select(1, b).select(1, inversePrePad).storageOffset() - 1
        val outputOffset = output.storageOffset() - 1
        j = 0
        while(j < channel * height * width) {
          paddedRatioData(paddedRatioOffset + j) = gradOutputData(gradOutputOffset + j) * outputData(outputOffset + j)
          paddedRatioData(paddedRatioOffset + j) /= scaleData(scaleOffset + j)
          j += 1
        }
        val accumRatioOffset = accumRatio.select(1, b).storageOffset() - 1
        j = 0
        while(j < height * width) {
          accumRatioData(accumRatioOffset + j) = 0
          j += 1
        }
        var c = 0
        val initPaddedRatioOffset = paddedRatio.select(1, b).storageOffset() - 1
        while(c < size - 1) {
          j = 0
          while(j < width * height) {
            accumRatioData(accumRatioOffset + j) += paddedRatioData(initPaddedRatioOffset + c * width * height + j)
            j += 1
          }
          c += 1
        }

        val accumRatioTimeInputOffset = accumRatioTimeInput.select(1, b).storageOffset() - 1
        val inputOffset = input.select(1, b).storageOffset() - 1
        c = 0
        while(c < channel) {
          j = 0
          while(j < height * width) {
            accumRatioData(accumRatioOffset + j) += paddedRatioData(initPaddedRatioOffset +
              (c + size - 1) * width * height + j)
            accumRationTimeInputData(accumRatioTimeInputOffset + j) =
              accumRatioData(accumRatioOffset + j) * inputData(inputOffset + c * height * width + j)
            gradInputData(gradInputOffset + c * height * width + j) -=
              ratioValue * accumRationTimeInputData(accumRatioTimeInputOffset + j)
            accumRatioData(accumRatioOffset + j) -=
              paddedRatioData(initPaddedRatioOffset + j + c * width * height)
            j += 1
          }
          c += 1
        }
      }
      i += 1
    }

    i = 0
    while(i < batchNum) {
      Await.result(results(i), Duration.Inf)
      i += 1
    }
  }

  private def LRNBackwardFloat(
     input : Tensor[Float], output : Tensor[Float], gradOutput : Tensor[Float],
     gradInput : Tensor[Float], paddedRatio : Tensor[Float], scale : Tensor[Float],
     accumRatio : Tensor[Float], accumRatioTimeInput : Tensor[Float],
     size : Int, alpha : Float, beta : Float, results : Array[Future[Unit]]) : Unit = {

    val batchNum = input.size(1)
    val channel = input.size(2)
    val height = input.size(3)
    val width = input.size(4)

    val paddedRatioData = paddedRatio.storage().array()
    val gradInputData = gradInput.storage().array()
    val gradOutputData = gradOutput.storage().array()
    val outputData = output.storage().array()
    val scaleData = scale.storage().array()
    val accumRatioData = accumRatio.storage().array()
    val accumRationTimeInputData = accumRatioTimeInput.storage().array()
    val inputData = input.storage().array()
    val ratioValue = 2.0f * alpha * beta / size
    val inversePrePad = size - (size + 1) / 2
    var i = 0
    while(i < batchNum) {
      val b = i + 1
      results(i) = Future {
        val gradInputOffset = gradInput.select(1, b).storageOffset() - 1
        val gradOutputOffset = gradOutput.select(1, b).storageOffset() - 1
        val scaleOffset = scale.select(1, b).storageOffset() - 1

        var j = 0
        while(j < channel * height * width) {
          gradInputData(gradInputOffset + j) = math.pow(scaleData(scaleOffset + j), -beta).toFloat
          gradInputData(gradInputOffset + j) *= gradOutputData(gradOutputOffset + j)
          j += 1
        }

        val initPaddedRatioOffset = paddedRatio.select(1, b).storageOffset() - 1
        val paddedRatioOffset = paddedRatio.select(1, b).select(1, inversePrePad).storageOffset() - 1
        val outputOffset = output.storageOffset() - 1
        j = 0
        while(j < channel * height * width) {
          paddedRatioData(paddedRatioOffset + j) = gradOutputData(gradOutputOffset + j) * outputData(outputOffset + j)
          paddedRatioData(paddedRatioOffset + j) /= scaleData(scaleOffset + j)
          j += 1
        }
        val accumRatioOffset = accumRatio.select(1, b).storageOffset() - 1
        j = 0
        while(j < height * width) {
          accumRatioData(accumRatioOffset + j) = 0
          j += 1
        }
        var c = 0
        while(c < size - 1) {
          j = 0
          while(j < width * height) {
            accumRatioData(accumRatioOffset + j) += paddedRatioData(initPaddedRatioOffset + c * width * height + j)
            j += 1
          }
          c += 1
        }

        val accumRatioTimeInputOffset = accumRatioTimeInput.select(1, b).storageOffset() - 1
        val inputOffset = input.select(1, b).storageOffset() - 1
        c = 0
        while(c < channel) {
          j = 0
          while(j < height * width) {
            accumRatioData(accumRatioOffset + j) += paddedRatioData(initPaddedRatioOffset +
              (c + size - 1) * width * height + j)
            accumRationTimeInputData(accumRatioTimeInputOffset + j) =
              accumRatioData(accumRatioOffset + j) * inputData(inputOffset + c * height * width + j)
            gradInputData(gradInputOffset + c * height * width + j) -=
              ratioValue * accumRationTimeInputData(accumRatioTimeInputOffset + j)
            accumRatioData(accumRatioOffset + j) -=
              paddedRatioData(initPaddedRatioOffset + j + c * width * height)
            j += 1
          }
          c += 1
        }
      }
      i += 1
    }

    i = 0
    while(i < batchNum) {
      Await.result(results(i), Duration.Inf)
      i += 1
    }
  }

  private def LRNForwardDouble(input : Tensor[Double], output : Tensor[Double], paddedSquare : Tensor[Double],
    scale : Tensor[Double], prePad : Int, alpha : Double, size : Int, beta : Double, k : Double,
    results : Array[Future[Unit]]) : Unit = {

    val batchNum = input.size(1)
    val channel = input.size(2)
    val height = input.size(3)
    val width = input.size(4)

    val outputData = output.storage().array()
    val inputData = input.storage().array()
    val paddedSquareData = paddedSquare.storage().array()
    val scaleData = scale.storage().array()

    var i = 0
    while(i < batchNum) {
      val b = i + 1
      results(i) = Future {
        // Square input
        val inputOffset = input.select(1, b).storageOffset() - 1
        val initPaddedSquareOffset = paddedSquare.select(1, b).select(1, prePad + 1).storageOffset() - 1
        var j = 0
        while(j < height * width * channel) {
          paddedSquareData(initPaddedSquareOffset + j) = inputData(inputOffset + j) * inputData(inputOffset + j)
          j += 1
        }

        // Init scale with k
        val scaleOffset = scale.select(1, b).storageOffset() - 1
        j = 0
        while(j < channel * height * width) {
          scaleData(scaleOffset + j) = k
          j += 1
        }

        // Sum first size of channels squared input data into first channel of scale
        val alphaOverSize = alpha / size
        val paddedSquareOffset = paddedSquare.select(1, b).storageOffset() - 1
        var c = 0
        while(c < size) {
          j = 0
          while(j < height * width) {
            scaleData(scaleOffset + j) += alphaOverSize * paddedSquareData(paddedSquareOffset + c * height * width + j)
            j += 1
          }
          c += 1
        }

        // Shift a window across the kernel
        c = 1
        while(c < channel) {
          System.arraycopy(scaleData, scaleOffset + (c - 1) * height * width, scaleData,
            scaleOffset + c * height * width, height * width)
          j = 0
           while(j < height * width) {
             scaleData(scaleOffset + c * height * width + j) += alphaOverSize *
                 paddedSquareData(paddedSquareOffset + (c + size -1) * height * width + j)
              scaleData(scaleOffset + c * height * width + j) -= alphaOverSize *
                  paddedSquareData(paddedSquareOffset + (c - 1) * height * width + j)
               j += 1
            }
          c += 1
        }

        // apply scale to input to get the output
        val outputOffset = output.select(1, b).storageOffset() - 1
        j = 0
        while(j < channel * height * width) {
          outputData(outputOffset + j) = math.pow(scaleData(scaleOffset + j), -beta) * inputData(inputOffset + j)
          j += 1
        }
      }
      i += 1
    }

    i = 0
    while(i < batchNum) {
      Await.result(results(i), Duration.Inf)
      i += 1
    }
  }

  private def LRNForwardFloat(input : Tensor[Float], output : Tensor[Float], paddedSquare : Tensor[Float],
                               scale : Tensor[Float], prePad : Int, alpha : Float, size : Int, beta : Float, k : Float,
                               results : Array[Future[Unit]]) : Unit = {

    val batchNum = input.size(1)
    val channel = input.size(2)
    val height = input.size(3)
    val width = input.size(4)

    val outputData = output.storage().array()
    val inputData = input.storage().array()
    val paddedSquareData = paddedSquare.storage().array()
    val scaleData = scale.storage().array()

    var i = 0
    while(i < batchNum) {
      val b = i + 1
      results(i) = Future {
        // Square input
        val inputOffset = input.select(1, b).storageOffset() - 1
        val initPaddedSquareOffset = paddedSquare.select(1, b).select(1, prePad + 1).storageOffset() - 1
        var j = 0
        while(j < height * width * channel) {
          paddedSquareData(initPaddedSquareOffset + j) = inputData(inputOffset + j) * inputData(inputOffset + j)
          j += 1
        }

        // Init scale with k
        val scaleOffset = scale.select(1, b).storageOffset() - 1
        j = 0
        while(j < channel * height * width) {
          scaleData(scaleOffset + j) = k
          j += 1
        }

        // Sum first size of channels squared input data into first channel of scale
        val alphaOverSize = alpha / size
        val paddedSquareOffset = paddedSquare.select(1, b).storageOffset() - 1
        var c = 0
        while(c < size) {
          j = 0
          while(j < height * width) {
            scaleData(scaleOffset + j) += alphaOverSize *
              paddedSquareData(paddedSquareOffset + c * height * width + j)
            j += 1
          }
          c += 1
        }

        // Shift a window across the kernel
        c = 1
        while(c < channel) {
          System.arraycopy(scaleData, scaleOffset + (c - 1) * height * width, scaleData,
            scaleOffset + c * height * width, height * width)
          j = 0
          while(j < height * width) {
            scaleData(scaleOffset + c * height * width + j) += alphaOverSize *
              paddedSquareData(paddedSquareOffset + (c + size -1) * height * width + j)
            scaleData(scaleOffset + c * height * width + j) -= alphaOverSize *
              paddedSquareData(paddedSquareOffset + (c - 1) * height * width + j)
            j += 1
          }
          c += 1
        }

        // apply scale to input to get the output
        val outputOffset = output.select(1, b).storageOffset() - 1
        j = 0
        while(j < channel * height * width) {
          outputData(outputOffset + j) = math.pow(scaleData(scaleOffset + j), -beta).toFloat * inputData(inputOffset + j)
          j += 1
        }
      }
      i += 1
    }

    i = 0
    while(i < batchNum) {
      Await.result(results(i), Duration.Inf)
      i += 1
    }
  }
}
