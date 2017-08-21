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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{DataConverter, ModuleData, ModuleSerializable, ModuleSerializer}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * Enable user stack multiple simple cells.
 */
class MultiCell[T : ClassTag](cells: Array[Cell[T]])(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {
  override def updateOutput(input: Table): Table = {
    var i = 0
    var result = input
    while (i < cells.length) {
      result = cells(i).forward(result)
      i += 1
    }

    this.output = result
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    var i = cells.length - 1
    var error = gradOutput
    while (i > 0) {
      val input = cells(i - 1).output
      error = cells(i).updateGradInput(input, error)
      i -= 1
    }
    error = cells(0).updateGradInput(input, error)

    this.gradInput = error
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    var i = cells.length - 1
    var currentModule = cells(i)
    var currentGradOutput = gradOutput
    while (i > 0) {
      val previousModule = cells(i - 1)
      currentModule.accGradParameters(previousModule.output, currentGradOutput)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
      i -= 1
    }

    currentModule.accGradParameters(input, currentGradOutput)
  }

  override def backward(input: Table, gradOutput: Table): Table = {
    var i = cells.length - 1
    var error = gradOutput
    while (i > 0) {
      val input = cells(i - 1).output
      error = cells(i).backward(input, error)
      i -= 1
    }
    error = cells(0).backward(input, error)

    this.gradInput = error
    gradInput
  }

}
