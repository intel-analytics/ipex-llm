package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SoftMax[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T]{

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    SoftMax.updateOutput[T](input, output)
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(output)
    SoftMax.updateGradInput[T](input, gradOutput, gradInput, output)
    gradInput
  }
}

object SoftMax{
  // Notice: SoftMin will call this function
  private[nn] def updateOutput[T: ClassTag](input: Tensor[T], output: Tensor[T])
    (implicit ev: TensorNumeric[T]) : Tensor[T] = {
    require(1 <= input.nDimension() && input.nDimension() <= 4, "1D, 2D, 3D or 4D tensor expected")
    val (nFrame, dim, stride) = if (input.nDimension() == 1) {
      (1, input.size(1), 1)
    } else if (input.nDimension() == 2) {
      (input.size(1), input.size(2), 1)
    } else if (input.nDimension() == 3) {
      (1, input.size(1), input.size(2) * input.size(3))
    } else {
      (input.size(1), input.size(2), input.size(3) * input.size(4))
    }

    val outputArray = output.storage().array()
    val inputArray = if (input.isContiguous()) {
      input.storage().array()
    } else {
      input.contiguous().storage().array()
    }

    var t = 0
    while (t < stride * nFrame) {
      val inputOffset = (t / stride) * dim * stride + t % stride
      val outputOffset = (t / stride) * dim * stride + t % stride

      var inputMax : T =  ev.fromType[Float](Float.MinValue)

      var d = 0
      while (d < dim) {
        if (ev.isGreater(inputArray(d * stride + inputOffset), inputMax)) {
          inputMax = inputArray(d * stride + inputOffset)
        }
        d += 1
      }

      var sum = ev.fromType[Int](0)
      d = 0
      while (d < dim) {
        val z = ev.exp(ev.minus(inputArray(d * stride + inputOffset), inputMax))
        outputArray(d * stride + outputOffset) = z
        sum = ev.plus(sum, z)
        d += 1
      }

      d = 0
      while (d < dim) {
        outputArray(d * stride + outputOffset) =
          ev.times(outputArray(d * stride + outputOffset), ev.divide(ev.fromType[Int](1), sum))
        d += 1
      }

      t += 1
    }

    output
  }

  private[nn] def updateGradInput[T: ClassTag](input: Tensor[T], gradOutput: Tensor[T],
    gradInput: Tensor[T], output: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(input.size().deep == gradOutput.size().deep)
    val (nFrame, dim, stride) = if (output.nDimension() == 1) {
      (1, output.size(1), 1)
    } else if (output.nDimension() == 2) {
      (output.size(1), output.size(2), 1)
    } else if (output.nDimension() == 3) {
      (1, output.size(1), output.size(2) * output.size(3))
    } else {
      (output.size(1), output.size(2), output.size(3) * output.size(4))
    }

    val gradInputArray = gradInput.storage().array()
    val outputArray = if (output.isContiguous()) {
      output.storage().array()
    } else {
      output.contiguous().storage().array()
    }
    val gradOutputArray = if (gradOutput.isContiguous()) {
      gradOutput.storage().array()
    } else {
      gradOutput.contiguous().storage().array()
    }

    var t = 0
    while (t < stride * nFrame) {
      val gradInputOffset = (t / stride) * dim * stride + t % stride
      val outputOffset = (t / stride) * dim * stride + t % stride
      val gradOutputOffset = (t / stride) * dim * stride + t % stride

      var sum = ev.fromType[Int](0)
      var d = 0
      while (d < dim) {
        sum = ev.plus(sum, ev.times(gradOutputArray(d * stride + gradOutputOffset),
          outputArray(d * stride + outputOffset)))
        d += 1
      }

      d = 0
      while (d < dim) {
        gradInputArray(d * stride + gradInputOffset) =
          ev.times(outputArray(d * stride + outputOffset),
            ev.minus(gradOutputArray(d * stride + gradOutputOffset), sum))
        d += 1
      }

      t += 1
    }

    gradInput
  }
}
