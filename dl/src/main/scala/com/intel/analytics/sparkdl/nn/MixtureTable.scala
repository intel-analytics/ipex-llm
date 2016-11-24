/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.utils.Table

import scala.reflect.ClassTag

/**
 * Creates a module that takes a table {gater, experts} as input and outputs the mixture of
 * experts (a Tensor or table of Tensors) using a gater Tensor. When dim is provided, it
 * specifies the dimension of the experts Tensor that will be interpolated (or mixed). Otherwise,
 * the experts should take the form of a table of Tensors. This Module works for experts of
 * dimension 1D or more, and for a 1D or 2D gater, i.e. for single examples or mini-batches.
 * @param dim
 * @tparam T Numeric type. Only support float/double now
 */
class MixtureTable[T: ClassTag](var dim: Int = Int.MaxValue)
 (implicit ev: TensorNumeric[T]) extends Module[Table, Tensor[T], T] {

  var size = Storage[Double]()//new ArrayBuffer[Int]()
  var size2 = Storage[Double]() //new ArrayBuffer[Int]()
  var batchSize = 0
  var backwardSetup = false

  var dimG: Int = 2


  // --buffer
  var _sum: Tensor[T] = Tensor[T]()
  var _expertView2 = Tensor[T]()
  var _expert2 = Tensor[T]()

  var _gaterView = Tensor[T]()
  var _expert = Tensor[T]()
  var _expertView = Tensor[T]()


  override def updateOutput(input: Table): Tensor[T] = {
    val gaterInput = input[Tensor[T]](1)
    //val expertInputs = input(2)
    var _batchSize = gaterInput.size(1)

    if (gaterInput.dim() < 2) {
      dimG = 1
      _batchSize = 1
      dim = if (dim == Int.MaxValue) 1 else dim
    }

    dim = if (dim == Int.MaxValue) 2 else dim

    if (input(2).isInstanceOf[Table]) {
      val expertInputs = input[Table](2)
      // -- expertInputs is a Table
      require(gaterInput.size(dimG) == expertInputs.length(),
        "Should be one gater output per expert")
      val _expertInput = expertInputs.asInstanceOf[Table](1).asInstanceOf[Tensor[T]]

      if (_batchSize != batchSize) {
        size.resize(_expertInput.dim() + 1).fill(1.0, 1, _expertInput.dim() + 1)
        if (dimG > 1) size(0) = gaterInput.size(1)
        size(dim - 1) = gaterInput.size(dimG)
        output.resizeAs(_expertInput)
        backwardSetup = false
        batchSize = _batchSize
      }
      // _gaterView.view(gaterInput, size) equal below??
      _gaterView = gaterInput.view(size.array().map(_.toInt))

      var i = 1
      while (i <= expertInputs.length()) {
        val _expertInput = expertInputs[Tensor[T]](i)
        val gate =  _gaterView.select(dim, i).expandAs(_expertInput)
        output.addcmul(_expertInput, gate)
        i += 1
      }
    } else if (input(2).isInstanceOf[Tensor[T]]) {
      val expertInputs = input[Tensor[T]](2)
      val _expertInput = expertInputs.asInstanceOf[Tensor[T]]
      if (_batchSize != batchSize) {
        size.resize(_expertInput.dim()).fill(1.0, 1, _expertInput.dim())
        if (dimG > 1) size(0) = gaterInput.size(1)
        size(dim - 1) = gaterInput.size(dimG)
        output.resizeAs(expertInputs.select(dim, 1))
        backwardSetup = false
        batchSize = _batchSize
      }
      _gaterView = gaterInput.view(size.array().map(_ .toInt))
      _expert.resizeAs(expertInputs).cmul(_gaterView.expandAs(expertInputs), expertInputs)
      output.sum(_expert, dim)
      output.resizeAs(expertInputs.select(dim, 1))
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val  gaterInput = input[Tensor[T]](1)
    //gradInput = Utils.recursiveResizeAs[T](gradInput.asInstanceOf[Table], input).toTable()
    if (!gradInput.contains(1)) gradInput.insert(1, Tensor[T]())
    val gaterGradInput = gradInput[Tensor[T]](1)
    // ??? val expertGradInputs = gradInput(2) no continue???

    if (input(2).isInstanceOf[Table]) {
      val expertInputs = input[Table](2)
      val expertGradInputs = new Table()
      if (!backwardSetup) {
        var i = 1
        while (i <= expertInputs.asInstanceOf[Table].length()) {
          val _expertInput = expertInputs.asInstanceOf[Table][Tensor[T]](i)
          val expertGradInput = if (expertGradInputs.isInstanceOf[Table] && expertGradInputs.asInstanceOf[Table].contains(i)) expertGradInputs.asInstanceOf[Table][Tensor[T]](i) else _expertInput.clone()
          expertGradInput.resizeAs(_expertInput)
          expertGradInputs.asInstanceOf[Table].insert(i, expertGradInput)
          i += 1
        }
        gaterGradInput.resizeAs(gaterInput)
        backwardSetup = true
      }

      var i = 1
      while (i <= expertGradInputs.asInstanceOf[Table].length()) {
        val _expertGradInput = expertGradInputs.asInstanceOf[Table][Tensor[T]](i)
        val _expertInput = expertInputs.asInstanceOf[Table][Tensor[T]](i)
        _expert.resizeAs(_expertInput).cmul(gradOutput, _expertInput)
        if (dimG == 1) {
          _expertView = _expert.view(_expert.nElement())
        } else {
          val tmp = _expert.nElement() / gradOutput.size(1)
         _expertView = _expert.view(gradOutput.size(1), tmp)
        }
        _sum.sum(_expertView, dimG)
        if (dimG == 1) {
          // to check
          gaterGradInput(Array(i)) = _sum(Array(dimG))
        } else {
          gaterGradInput.select(dimG,i).copy(_sum.select(dimG,1))
        }

        // -- expert updateGradInput
        val gate =_gaterView.select(dim,i).expandAs(_expertGradInput)
        _expertGradInput.cmul(gate, gradOutput)
        i += 1
      }

      gradInput.insert(2, expertGradInputs)

    } else if (input(2).isInstanceOf[Tensor[T]]){
      val expertInputs = input[Tensor[T]](2)
      val expertGradInputs = Tensor[T]()
      if (!backwardSetup) {
        size2.resize(expertInputs.dim())
        size2.copy(Storage(expertInputs.size().map(_.toDouble)))
        size2(dim - 1) = 1
        gaterGradInput.resizeAs(gaterInput)
        backwardSetup = true
      }

      // -- gater updateGradInput
      _expertView = gradOutput.view(size2.array().map(_.toInt))
      val  _gradOutput = _expertView.expandAs(expertInputs)
      _expert.resizeAs(expertInputs).cmul(_expertView.expandAs(expertInputs), expertInputs)
      var expert = _expert.transpose(dim, dimG)
      if ( !expert.isContiguous()) {
       _expert2.resizeAs(expert)
       _expert2.copy(expert)
        expert = _expert2
      }

      if (dimG == 1) {
        val tmp = expert.nElement() / (gaterInput.size(1))
       _expertView2 = expert.view(gaterInput.size(1), expert.nElement())
      } else {
        val tmp = expert.nElement() / (gaterInput.size(1) * gaterInput.size(2))
       _expertView2 = expert.view(gaterInput.size(1), gaterInput.size(2), tmp)
      }
      gaterGradInput.sum(_expertView2, dimG + 1)
      gaterGradInput.resizeAs(gaterInput)

      // -- expert updateGradInput
      expertGradInputs.resizeAs(expertInputs).cmul(_gaterView.expandAs(expertInputs), _gradOutput)
      gradInput.insert(2, expertGradInputs)
    }
    gradInput
  }
}
