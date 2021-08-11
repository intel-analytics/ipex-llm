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

package com.intel.analytics.bigdl.tensor

object SparseTensorApply {

  /**
   * Iterate through (sparse) tensor1, and apply func to the elements,
   * set function result to (sparse) tensor 2
   *
   * @param tensor1 the tensor1
   * @param tensor2 the result tensor
   * @param func    (tensor1Data, tensor1.StorageOffset, tensor2Data,
   *                tensor2.StorageOffset)
   */
  def apply1[A, B](tensor1: Tensor[A], tensor2: Tensor[B],
    func: TensorDiffTypeFunc4[A, B]): Unit = {

    require(tensor1.getTensorType == SparseType,
      s"Wrong TensorType found at tensor1: ${tensor1.getTensorType}")
    require(tensor2.getTensorType == SparseType,
      s"Wrong TensorType found at tensor2: ${tensor2.getTensorType}")

    val t1 = tensor1.asInstanceOf[SparseTensor[A]]
    val t2 = tensor2.asInstanceOf[SparseTensor[B]]
    require(t1._nElement == t2._nElement,
      s"nElement of tensor1(${t1._nElement}) is't equal to nElement of tensor2(${t2._nElement})")

    val array1 = t1.storage().array()
    val array2 = t2.storage().array()
    var i = 0
    while (i < t1._nElement) {
      func(array1, t1._storageOffset + i, array2, t2._storageOffset + i)
      i += 1
    }
  }

}
