package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.Tensor
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SpatialZeroPadding[@specialized(Float, Double) T: ClassTag](padLeft : Int, padRight : Int, padTop : Int, padBottom : Int)(implicit ev: TensorNumeric[T]) extends Module[T] {
  def this(padLeft : Int)(implicit ev: TensorNumeric[T]) = this(padLeft, padLeft, padLeft, padLeft)

  override def updateOutput(input : Tensor[T]) = {
    if(input.dim() == 3) {
      // sizes
      val h = input.size(2) + this.padTop + this.padBottom
      val w = input.size(3) + this.padLeft + this.padRight
      if(w < 1 || h < 1) {
        throw new IllegalArgumentException("input is too small")
      }
      this.output.resize(Array(input.size(1), h, w))
      this.output.zero()

      // crop input if necessary
      var cInput = input
      if(this.padTop < 0) cInput = cInput.narrow(2, 1 - this.padTop, cInput.size(2) + this.padTop)
      if(this.padBottom < 0) cInput = cInput.narrow(2, 1, cInput.size(2) + this.padBottom)
      if(this.padLeft < 0) cInput = cInput.narrow(3, 1 - this.padLeft, cInput.size(3) + this.padLeft)
      if(this.padRight < 0) cInput = cInput.narrow(3, 1, cInput.size(3) + this.padRight)

      // crop output if necessary
      var cOutput = output
      if(this.padTop > 0) cOutput = cOutput.narrow(2, 1 + this.padTop, cOutput.size(2) - this.padTop)
      if(this.padBottom > 0) cOutput = cOutput.narrow(2, 1, cOutput.size(2) - this.padBottom)
      if(this.padLeft > 0) cOutput = cOutput.narrow(3, 1 + this.padLeft, cOutput.size(3) - this.padLeft)
      if(this.padRight > 0) cOutput = cOutput.narrow(3, 1, cOutput.size(3) - this.padRight)

      cOutput.copy(cInput)
    } else if(input.dim() == 4) {
      // sizes
      val h = input.size(3) + this.padTop + this.padBottom
      val w = input.size(4) + this.padLeft + this.padRight
      if(w < 1 || h < 1) {
        throw new IllegalArgumentException("input is too small")
      }
      this.output.resize(Array(input.size(1), input.size(2), h, w))
      this.output.zero()

      // crop input if necessary
      var cInput = input
      if(this.padTop < 0) cInput = cInput.narrow(3, 1 - this.padTop, cInput.size(3) + this.padTop)
      if(this.padBottom < 0) cInput = cInput.narrow(3, 1, cInput.size(3) + this.padBottom)
      if(this.padLeft < 0) cInput = cInput.narrow(4, 1 - this.padLeft, cInput.size(4) + this.padLeft)
      if(this.padRight < 0) cInput = cInput.narrow(4, 1, cInput.size(4) + this.padRight)

      // crop output if necessary
      var cOutput = output
      if(this.padTop > 0) cOutput = cOutput.narrow(3, 1 + this.padTop, cOutput.size(3) - this.padTop)
      if(this.padBottom > 0) cOutput = cOutput.narrow(3, 1, cOutput.size(3) - this.padBottom)
      if(this.padLeft > 0) cOutput = cOutput.narrow(4, 1 + this.padLeft, cOutput.size(4) - this.padLeft)
      if(this.padRight > 0) cOutput = cOutput.narrow(4, 1, cOutput.size(4) - this.padRight)

      cOutput.copy(cInput)
    } else {
      throw new IllegalArgumentException("input must be 3 or 4-dimensional")
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]) : Tensor[T] = {
    if(input.dim() == 3) {
      this.gradInput.resizeAs(input).zero()

      // crop gradInput if necessary
      var cgInput = gradInput
      if(this.padTop < 0) cgInput = cgInput.narrow(2, 1 - this.padTop, cgInput.size(2) + this.padTop)
      if(this.padBottom < 0) cgInput = cgInput.narrow(2, 1, cgInput.size(2) + this.padBottom)
      if(this.padLeft < 0) cgInput = cgInput.narrow(3, 1 - this.padLeft, cgInput.size(3) + this.padLeft)
      if(this.padRight < 0) cgInput = cgInput.narrow(3, 1, cgInput.size(3) + this.padRight)

      // crop output if necessary
      var cgOutput = gradOutput
      if(this.padTop > 0) cgOutput = cgOutput.narrow(2, 1 + this.padTop, cgOutput.size(2) - this.padTop)
      if(this.padBottom > 0) cgOutput = cgOutput.narrow(2, 1, cgOutput.size(2) - this.padBottom)
      if(this.padLeft > 0) cgOutput = cgOutput.narrow(3, 1 + this.padLeft, cgOutput.size(3) - this.padLeft)
      if(this.padRight > 0) cgOutput = cgOutput.narrow(3, 1, cgOutput.size(3) - this.padRight)

      cgInput.copy(cgOutput)
    } else if(input.dim() == 4) {
      this.gradInput.resizeAs(input).zero()

      // crop gradInput if necessary
      var cgInput = gradInput
      if(this.padTop < 0) cgInput = cgInput.narrow(3, 1 - this.padTop, cgInput.size(3) + this.padTop)
      if(this.padBottom < 0) cgInput = cgInput.narrow(3, 1, cgInput.size(3) + this.padBottom)
      if(this.padLeft < 0) cgInput = cgInput.narrow(4, 1 - this.padLeft, cgInput.size(4) + this.padLeft)
      if(this.padRight < 0) cgInput = cgInput.narrow(4, 1, cgInput.size(4) + this.padRight)

      // crop output if necessary
      var cgOutput = gradOutput
      if(this.padTop > 0) cgOutput = cgOutput.narrow(3, 1 + this.padTop, cgOutput.size(3) - this.padTop)
      if(this.padBottom > 0) cgOutput = cgOutput.narrow(3, 1, cgOutput.size(3) - this.padBottom)
      if(this.padLeft > 0) cgOutput = cgOutput.narrow(4, 1 + this.padLeft, cgOutput.size(4) - this.padLeft)
      if(this.padRight > 0) cgOutput = cgOutput.narrow(4, 1, cgOutput.size(4) - this.padRight)

      cgInput.copy(cgOutput)
    } else {
      throw new IllegalArgumentException("input must be 3 or 4-dimensional")
    }

    this.gradInput
  }

  override def toString() : String = {
    s"nn.SpatialZeroPadding(l=$padLeft, r=$padRight, t=$padTop, b=$padBottom)"
  }
}
