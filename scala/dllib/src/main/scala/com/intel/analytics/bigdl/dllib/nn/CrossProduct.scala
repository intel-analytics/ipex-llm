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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * A layer which takes a table of multiple tensors(n >= 2) as input
 * and calculate to dot product for `all combinations of pairs` among input tensors.
 * <br><br>
 * Dot-product outputs are ordered according to orders of pairs in input Table.
 * For instance, input (Table) is T(A, B, C), output (Tensor) will be [A.*B, A.*C, B.*C].
 * <br><br>
 * Dimensions of input' Tensors could be one or two, if two, first dimension is `batchSize`.
 * For convenience, output is 2-dim Tensor regardless of input' dims.
 * <br><br>
 * Table size checking and Tensor size checking will be execute before each forward,
 * when [[numTensor]] and [[embeddingSize]] are set values greater than zero.
 *
 * @param numTensor (for checking)number of Tensor input Table contains, default: 0(won't check)
 * @param embeddingSize (for checking)vector length of dot product, default: 0(won't check)
 */
class CrossProduct[T: ClassTag](
  val numTensor: Int = 0,
  val embeddingSize: Int = 0
)(implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
    val len = input.length()
    require(numTensor <= 0 || numTensor == len,
      s"Input tensor number is $len, unequal to numTensor($numTensor)!")

    val (_, batch, _) = getShape(input[Tensor[T]](1))
    output.resize(batch, len * (len - 1) / 2)

    if (embeddingSize > 0) {
      var i = 1
      while (i <= len) {
        checkEmbeddingSize(input(i))
        i += 1
      }
    }

    var cc = 1
    var i = 1
    var j = 2
    while (i < len) {
      val ijDot = batchDot(input(i), input(j))
      output.select(2, cc).copy(ijDot)

      cc += 1
      if (j == len) {
        i += 1
        j = i + 1
      } else {
        j += 1
      }
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput = T()

    val len = input.length()
    val gout = gradOutput

    require(gout.dim() == 2, s"invalid dim of gradOutput(${gout.dim()})!")

    val outLen = len * (len - 1) / 2
    require(gout.size(2) == outLen,
      s"invalid colSize of gradOutput(${gout.size(2)}), it should be $outLen!")

    val (dim, _, emLen) = getShape(input[Tensor[T]](1))

    var cc = 1
    var i = 1
    var j = 2
    while (i < len) {
      val (ti, tj) = dim match {
        case 1 =>
          input[Tensor[T]](i).view(1, emLen) -> input[Tensor[T]](j).view(1, emLen)
        case 2 =>
          input[Tensor[T]](i) -> input[Tensor[T]](j)
      }

      // get cc_th column data from total gradOut
      val go = gout.narrow(2, cc, 1)

      val jInc = Tensor[T]().resizeAs(ti).copy(ti).cmul(go)
      if (dim == 1) jInc.squeeze()
      gradInput.get[Tensor[T]](j) match {
        case None => gradInput.update(j, jInc)
        case Some(v) => v.add(jInc)
      }

      val iInc = Tensor[T]().resizeAs(tj).copy(tj).cmul(go)
      if (dim == 1) iInc.squeeze()
      gradInput.get[Tensor[T]](i) match {
        case None => gradInput.update(i, iInc)
        case Some(v) => v.add(iInc)
      }

      cc += 1
      if (j == len) {
        i += 1
        j = i + 1
      } else {
        j += 1
      }
    }

    gradInput
  }

  protected def checkEmbeddingSize(t: Tensor[T]): Unit = {
    val size = if (t.dim() == 1) t.size(1) else t.size(2)
    require(embeddingSize <= 0 || embeddingSize == size,
      s"size of input Tensor($size) not equal to embeddingSize($embeddingSize)!")
  }

  protected def batchDot(t1: Tensor[T], t2: Tensor[T]): Tensor[T] = {
    var (input1, input2) = (t1, t2)

    if (input1.dim() == 1) {
      input1 = input1.view(1, input1.size(1))
      input2 = input2.view(1, input2.size(1))
    }

    val buffer = Tensor[T]()
    buffer.resizeAs(input1).cmul(input1, input2)
    buffer.sum(2).squeeze()
  }

  private def getShape(t: Tensor[T]) = {
    val (batch, size) = t.dim() match {
      case 1 => 1 -> t.size(1)
      case 2 => t.size(1) -> t.size(2)
      case n => throw new IllegalArgumentException(s"wrong dim of input Tensor($n)!")
    }
    (t.dim(), batch, size)
  }

}

object CrossProduct {

  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): CrossProduct[T] = new CrossProduct[T]()

  def apply[T: ClassTag](
    numTensor: Int = 0,
    embeddingSize: Int = 0
  )(implicit ev: TensorNumeric[T]): CrossProduct[T] = {
    new CrossProduct(numTensor, embeddingSize)
  }

}
