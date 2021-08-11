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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Creates a module that takes a table {gater, experts} as input and outputs the mixture of
 * experts (a Tensor or table of Tensors) using a gater Tensor. When dim is provided, it
 * specifies the dimension of the experts Tensor that will be interpolated (or mixed). Otherwise,
 * the experts should take the form of a table of Tensors. This Module works for experts of
 * dimension 1D or more, and for a 1D or 2D gater, i.e. for single examples or mini-batches.
 *
 * @param dim
 * @tparam T Numeric type. Only support float/double now
 */

@SerialVersionUID( - 114773362363268868L)
class MixtureTable[T: ClassTag](var dim: Int = Int.MaxValue)
 (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  var size = Storage[Double]()
  var batchSize = 0
  var backwardSetup = false
  var dimG = 2

  // buffer
  private var expertView2 = Tensor[T]()
  private var gaterView = Tensor[T]()
  private val expert = Tensor[T]()
  private var expertView = Tensor[T]()

  override def updateOutput(input: Table): Tensor[T] = {
    val gaterInput = input[Tensor[T]](1)
    var inputBatchSize = gaterInput.size(1)

    if (gaterInput.dim() < 2) {
      dimG = 1
      inputBatchSize = 1
      if (dim == Int.MaxValue) dim = 1
    }
    if (dim == Int.MaxValue) dim = 2

    if (input(2).isInstanceOf[Table]) {
      // expertInputs is a Table
      val expertInputs = input[Table](2)
      val expertInput = expertInputs[Tensor[T]](1)
      require(gaterInput.size(dimG) == expertInputs.length(),
        "MixtureTable: Should one gater output per expert," +
          s" gater ${gaterInput.size(dimG)}, expert ${expertInputs.length()}")
      if (inputBatchSize != batchSize) {
        size.resize(expertInput.dim() + 1).fill(1.0, 1, expertInput.dim() + 1)
        if (dimG > 1) size(0) = gaterInput.size(1)
        size(dim - 1) = gaterInput.size(dimG)
        output.resizeAs(expertInput)
        backwardSetup = false
        batchSize = inputBatchSize
      }
      gaterView = gaterInput.view(size.array().map(_.toInt))

      var i = 1
      while (i <= expertInputs.length()) {
        val expertInput = expertInputs[Tensor[T]](i)
        output.addcmul(expertInput, gaterView.select(dim, i).expandAs(expertInput))
        i += 1
      }
    } else if (input(2).isInstanceOf[Tensor[T]]) {
      val expertInputs = input[Tensor[T]](2)
      if (inputBatchSize != batchSize) {
        size.resize(expertInputs.dim()).fill(1.0, 1, expertInputs.dim())
        if (dimG > 1) size(0) = gaterInput.size(1)
        size(dim - 1) = gaterInput.size(dimG)
        output.resizeAs(expertInputs.select(dim, 1))
        backwardSetup = false
        batchSize = inputBatchSize
      }
      gaterView = gaterInput.view(size.array().map(_ .toInt))
      expert.resizeAs(expertInputs).cmul(gaterView.expandAs(expertInputs), expertInputs)
      output.sum(expert, dim)
      output.resizeAs(expertInputs.select(dim, 1))
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val  gaterInput = input[Tensor[T]](1)
    if (!gradInput.contains(1)) gradInput.insert(1, Tensor[T]())

    if (input(2).isInstanceOf[Table]) {

      if (!gradInput.contains(2)) gradInput.insert(2, T())
      gradInput = Utils.recursiveResizeAs[T](gradInput, input).toTable

      val expertInputs = input[Table](2)
      val gaterGradInput = gradInput[Tensor[T]](1)
      val expertGradInputs = gradInput[Table](2)
      if (!backwardSetup) {
        var i = 1
        while (i <= expertInputs.length()) {
          val expertInput = expertInputs[Tensor[T]](i)
          val expertGradInput = if (expertGradInputs.contains(i)) {
            expertGradInputs[Tensor[T]](i)
          } else expertInput.clone()
          expertGradInput.resizeAs(expertInput)
          expertGradInputs.update(i, expertGradInput)
          i += 1
        }
        gaterGradInput.resizeAs(gaterInput)
        backwardSetup = true
      }

      var i = 1
      while (i <= expertGradInputs.length()) {
        val expertGradInput = expertGradInputs[Tensor[T]](i)
        val expertInput = expertInputs[Tensor[T]](i)
        expert.resizeAs(expertInput).cmul(gradOutput, expertInput)
        if (dimG == 1) {
          expertView = expert.view(expert.nElement())
        } else {
         expertView = expert.view(gradOutput.size(1), expert.nElement() / gradOutput.size(1))
        }
        expertView2.sum(expertView, dimG)
        if (dimG == 1) {
          gaterGradInput(Array(i)) = expertView2(Array(dimG))
        } else {
          gaterGradInput.select(dimG, i).copy(expertView2.select(dimG, 1))
        }

        // expert updateGradInput
        expertGradInput.cmul(gaterView.select(dim, i).expandAs(expertGradInput), gradOutput)
        i += 1
      }
    } else if (input(2).isInstanceOf[Tensor[T]]) {

      if (!gradInput.contains(2)) gradInput.insert(2, T())
      gradInput = Utils.recursiveResizeAs[T](gradInput, input).toTable

      val expertInputs = input[Tensor[T]](2)
      val gaterGradInput = gradInput[Tensor[T]](1)
      val expertGradInputs = gradInput[Tensor[T]](2)
      if (!backwardSetup) {
        size.resize(expertInputs.dim()).copy(Storage(expertInputs.size().map(_.toDouble)))
        size(dim - 1) = 1
        gaterGradInput.resizeAs(gaterInput)
        backwardSetup = true
      }

      // gater updateGradInput
      expertView = gradOutput.view(size.array().map(_.toInt))
      expertView.expandAs(expertInputs)
      expert.resizeAs(expertInputs).cmul(expertView, expertInputs)
      val expertC = expert.transpose(dim, dimG).contiguous()
      if (dimG == 1) {
       expertView2 = expertC.view(gaterInput.size(1), expertC.nElement() / (gaterInput.size(1)))
      } else {
       expertView2 = expertC.view(gaterInput.size(1), gaterInput.size(2),
         expertC.nElement() / (gaterInput.size(1) * gaterInput.size(2)))
      }
      gaterGradInput.sum(expertView2, dimG + 1)
      gaterGradInput.resizeAs(gaterInput)

      // expert updateGradInput
      expertGradInputs.resizeAs(expertInputs).cmul(gaterView.expandAs(expertInputs), expertView)
    }
    gradInput
  }

  override def clearState() : this.type = {
    super.clearState()
    expertView2.set()
    gaterView.set()
    expert.set()
    expertView.set()
    this
  }

  override def toString(): String = {
    s"${getPrintName}($dim)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[MixtureTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: MixtureTable[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        dim == that.dim
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), dim)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object MixtureTable {
  def apply[@specialized(Float, Double) T: ClassTag](dim: Int = Int.MaxValue)
      (implicit ev: TensorNumeric[T]) : MixtureTable[T] = {
    new MixtureTable[T](dim)
  }
}
