package com.intel.webscaleml.nn.nn.mkl_dnn

import com.intel.webscaleml.nn.mkl.Primitives
import com.intel.webscaleml.nn.nn.Module
import com.intel.webscaleml.nn.tensor.Tensor
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class ReLU[@specialized(Float, Double) T: ClassTag]()(implicit ev: TensorNumeric[T]) extends Module[T]{
  override def toString() : String = {
    s"nn.mkl_dnn.ReLU"
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(gradOutput)
    gradInput.copy(gradOutput)
    ev.getType() match {
      case "Float" => Primitives.relu_backward(input.storage().array().asInstanceOf[Array[Float]],
        gradOutput.storage().array().asInstanceOf[Array[Float]], gradInput.storage().array().asInstanceOf[Array[Float]],
        gradInput.size(4), gradInput.size(3), gradInput.size(2), gradInput.size(1))
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
    gradInput
  }

  override  def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    ev.getType() match {
      case "Float" => Primitives.relu_forward(input.storage().array().asInstanceOf[Array[Float]],
        output.storage().array().asInstanceOf[Array[Float]], input.size(4), input.size(3), input.size(2), input.size(1))
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
    output
  }
}
