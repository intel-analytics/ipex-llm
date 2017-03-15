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
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * hidden sizes in the Cell, whose length is the number of hiddens.
 * The elements correspond to the hidden sizes of returned hiddens
 *
 * E.g. For RnnCell, it should be Array(hiddenSize)
 *      For LSTM, it should be Array(hiddenSize, hiddenSize)
 *     (because each time step a LSTM return two hiddens `h` and `c` in order,
 *     which have the same size.)
 */
abstract class Cell[T : ClassTag](val hiddensShape: Array[Int])
  (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {

  /**
   * resize the hidden parameters wrt the batch size, hiddens shapes.
   *
   * e.g. RnnCell contains 1 hidden parameter (H), thus it will return Tensor(size)
   *      LSTM contains 2 hidden parameters (C and H) and will return T(Tensor(), Tensor())\
   *      and recursively intialize all the tensors in the Table.
   *
   * @param hidden
   * @param size batchSize
   * @return
   */
  def hidResize(hidden: Activity, size: Int): Activity = {
    if (hidden == null) {
      if (hiddensShape.length == 1) {
        hidResize(Tensor[T](), size)
      } else {
        val _hidden = T()
        var i = 1
        while (i <= hiddensShape.length) {
          _hidden(i) = Tensor[T]()
          i += 1
        }
        hidResize(_hidden, size)
      }
    } else {
      if (hidden.isInstanceOf[Tensor[T]]) {
        require(hidden.isInstanceOf[Tensor[T]],
          "Cell: hidden should be a Tensor")
        hidden.toTensor.resize(size, hiddensShape(0))
      } else {
        require(hidden.isInstanceOf[Table],
          "Cell: hidden should be a Table")
        var i = 1
        while (i <= hidden.toTable.length()) {
          hidden.toTable[Tensor[T]](i).resize(size, hiddensShape(i - 1))
          i += 1
        }
        hidden
      }
    }
  }
}
