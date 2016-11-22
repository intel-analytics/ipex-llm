package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.Table
import com.intel.analytics.sparkdl.tensor.Tensor
import jdk.internal.util.xml.impl.Input

import scala.reflect.ClassTag

class MixtureTable[T: ClassTag](var dim: Int = Int.MaxValue)
 (implicit ev: TensorNumeric[T]) extends Module[Table, Table, T] {

  val size = Tensor[T]()
  val sizes = Tensor[T]()
  var batchSize = 0
  val backwardSetup = false
  override def updateOutput(input: Table): Table = {
    val gaterInput = input[Tensor[T]](1)
    val expertInputs = input(2)

    val _gaterView = Tensor[T]()
    var _expert = Tensor[T]()
    var expertView = Tensor[T]()

    var dimG = 2
    var batchSize = gaterInput.size(1)

    if (gaterInput.dim() < 2) {
      dimG = 1
      batchSize = 1
      dim = if (dim != Int.MaxValue) 1 else dim
    }

    dim = if (dim != Int.MaxValue) 2 else dim

    if (expertInputs.isInstanceOf[Table]) {
      require(gaterInput.size(dimG) == expertInputs.asInstanceOf[Table].length(), "Should be one gater output per expert")
      

    } else if (expertInputs.isInstanceOf[Tensor[T]]) {

    }
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {

  }
}
