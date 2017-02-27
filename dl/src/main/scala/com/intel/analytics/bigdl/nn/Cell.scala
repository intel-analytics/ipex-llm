/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

abstract class Cell[T : ClassTag] ()
  (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {

  // number of hidden parameters in the Cell. E.g. one for RnnCell, two for LSTM.
  val nHids: Int

  /**
   * resize the hidden parameters wrt the size1, size2.
   * E.g. size1 = BatchSize, size2 = HiddenSize
   *
   * e.g. RnnCell contains 1 hidden parameter (H), thus it will return Tensor(size)
   *      LSTM contains 2 hidden parameters (C and H) and will return T(Tensor(), Tensor())\
   *      and recursively intialize all the tensors in the Table.
   *
   * @param hidden
   * @param size1 batchSize
   * @param size2 hiddenSize
   * @return
   */
  def hidResize(hidden: Activity, size1: Int, size2: Int): Activity = {
    if (hidden == null) {
      if (nHids == 1) {
        hidResize(Tensor[T](), size1, size2)
      } else {
        val _hidden = T()
        var i = 1
        while (i <= nHids) {
          _hidden(i) = Tensor[T]()
          i += 1
        }
        hidResize(_hidden, size1, size2)
      }
    } else {
      if (hidden.isInstanceOf[Tensor[T]]) {
        require(hidden.isInstanceOf[Tensor[T]],
          "Cell: hidden should be a Tensor")
        hidden.toTensor.resize(size1, size2)
        hidden.toTensor
      } else {
        require(hidden.isInstanceOf[Table],
          "Cell: hidden should be a Table")
        var i = 1
        while (i <= hidden.toTable.length()) {
          hidResize(hidden.toTable(i), size1, size2)
          i += 1
        }
        hidden
      }
    }
  }

}
