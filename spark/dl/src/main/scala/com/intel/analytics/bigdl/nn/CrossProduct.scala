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
 *
 */
class CrossProduct[T: ClassTag](
  isBackProp: Boolean = true,
  numTensor: Option[Int] = None,
  embeddingSize: Option[Int] = None
)(implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
    val len = input.length()
    require(numTensor.isEmpty || numTensor.get == len,
      s"len of input Table($len) not equal to numTensor(${numTensor.get})!")

    val (_, batch, _) = getShape(input[Tensor[T]](1))
    output.resize(batch, len * (len - 1) / 2)

    if (embeddingSize.nonEmpty) {
      (1 to len).foreach(i => checkEmbeddingSize(input(i)))
    }

    var cc = 1
    for( i <- 1 until len; j <- i + 1 to len ) {
      val ijDot = dot(input(i), input(j))
      output.select(2, cc).copy(ijDot)
      cc += 1
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput = T()

    if (isBackProp) {
      val (len, gout) = input.length() -> gradOutput
      require(gout.dim() == 2, s"invalid dim of gradOutput(${gout.dim()})!")

      val outLen = len * (len - 1) / 2
      require(gout.size(2) == outLen,
        s"invalid colSize of gradOutput(${gout.size(2)}), it should be $outLen!")

      val (dim, _, emLen) = getShape(input[Tensor[T]](1))

      var cc = 1
      for( i <- 1 until len; j <- i + 1 to len ) {
        val (ti, tj) = dim match {
          case 1 =>
            input[Tensor[T]](i).view(1, emLen) -> input[Tensor[T]](j).view(1, emLen)
          case 2 =>
            input[Tensor[T]](i) -> input[Tensor[T]](j)
        }

        val go = gout.narrow(2, cc, 1)

        val jInc = Tensor[T]().resizeAs(ti).copy(ti).cmul(go)
        if (dim == 1) jInc.resize(Array(jInc.size(2)))
        gradInput.get[Tensor[T]](j) match {
          case None => gradInput.update(j, jInc)
          case Some(v) => v.add(jInc)
        }

        val iInc = Tensor[T]().resizeAs(tj).copy(tj).cmul(go)
        if (dim == 1) iInc.resize(Array(iInc.size(2)))
        gradInput.get[Tensor[T]](i) match {
          case None => gradInput.update(i, iInc)
          case Some(v) => v.add(iInc)
        }

        cc += 1
      }
    }

    gradInput
  }

  protected def checkEmbeddingSize(t: Tensor[T]): Unit = {
    val size = if (t.dim() == 1) t.size(1) else t.size(2)
    require(embeddingSize.isEmpty || embeddingSize.get == size,
      s"size of input Tensor($size) not equal to embeddingSize(${embeddingSize.get})!")
  }

  protected def dot(t1: Tensor[T], t2: Tensor[T]): Tensor[T] = {
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
    backProp: Boolean = true,
    numTensor: Option[Int] = None,
    embeddingSize: Option[Int] = None
  )(implicit ev: TensorNumeric[T]): CrossProduct[T] = {
    new CrossProduct(backProp, numTensor, embeddingSize)
  }
}

