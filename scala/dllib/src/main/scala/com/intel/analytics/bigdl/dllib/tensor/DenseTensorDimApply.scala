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

object DenseTensorDimApply {
  private[tensor] def dimApply2[@specialized(Float, Double) T](tensor1: DenseTensor[T],
    tensor2: Tensor[T], _dim: Int, func: (Array[T], Int,
      Int, Int, Array[T], Int, Int, Int) => Unit): Unit = {

    require(_dim >= 0 && _dim < tensor1.nDimension, "invalid dimension")
    require(tensor1.nDimension == tensor2.nDimension(), "inconsistent tensor sizes")
    var i = 0
    while (i < tensor1.nDimension) {
      if (i != _dim) {
        require(tensor1.size(i + 1) == tensor2.size(i + 1), "inconsistent tensor sizes")
      }
      i += 1
    }

    val counter = new Array[Int](tensor1.nDimension)
    val _data1 = tensor1.storage().asInstanceOf[Storage[T]].array()
    var _offset1 = tensor1.storageOffset() - 1
    val stride1 = tensor1._stride(_dim)
    val size1 = tensor1._size(_dim)

    val _data2 = tensor2.storage().asInstanceOf[Storage[T]].array()
    var _offset2 = tensor2.storageOffset() - 1
    val stride2 = tensor2.stride(_dim + 1)
    val size2 = tensor2.size(_dim + 1)

    var hasFinished = false
    while (!hasFinished) {
      func(_data1, _offset1, stride1, size1,
        _data2, _offset2, stride2, size2)
      if (tensor1.nDimension == 1) {
        hasFinished = true
      } else {
        var i = 0
        var break = false
        while (i < tensor1.nDimension && !break) {
          if (i == _dim) {
            if (i == tensor1.nDimension - 1) {
              hasFinished = true
              break = true
            }
          } else {
            counter(i) += 1
            _offset1 += tensor1.stride(i + 1)
            _offset2 += tensor2.stride(i + 1)

            if (counter(i) == tensor1.size(i + 1)) {
              if (i == tensor1.nDimension - 1) {
                break = true
                hasFinished = true
              } else {
                _offset1 -= counter(i) * tensor1.stride(i + 1)
                _offset2 -= counter(i) * tensor2.stride(i + 1)
                counter(i) = 0
              }
            } else {
              break = true
            } // if (counter(i) == tensor1.size(i))
          } // if (i == _dim) else
          i += 1
        } // while
      } // if(tensor1.nDimension == 1)
    } // while(!hasFinished)
  }


  private[tensor] def dimApply3[@specialized(Float, Double) T](tensor1: DenseTensor[T],
    tensor2: Tensor[T], tensor3: Tensor[T], dim: Int,
    func: (
      Array[T], Int, Int, Int,
        Array[T], Int, Int, Int,
        Array[T], Int, Int, Int) => Unit): Unit = {
    require(dim > 0 && dim <= tensor1.nDimension, "invalid dimension")
    require(tensor1.nDimension == tensor2.nDimension, "inconsistent tensor sizes")
    require(tensor2.nDimension == tensor3.nDimension, "inconsistent tensor sizes")
    var d = 1
    while (d <= tensor1.nDimension) {
      if (d != dim) {
        require(tensor1.size(d) == tensor2.size(d), "inconsistent tensor sizes")
        require(tensor2.size(d) == tensor3.size(d), "inconsistent tensor sizes")
      }
      d += 1
    }

    val counter = new Array[Int](tensor1.nDimension)
    //    val _data1 = tensor1.storage().asInstanceOf[Storage[T]].array()
    val _data1 = tensor1.storage().array()
    var _offset1 = tensor1.storageOffset() - 1
    val stride1 = tensor1.stride(dim)
    val size1 = tensor1.size(dim)

    val _data2 = tensor2.storage().array()
    var _offset2 = tensor2.storageOffset() - 1
    val stride2 = tensor2.stride(dim)
    val size2 = tensor2.size(dim)

    val _data3 = tensor3.storage().array()
    var _offset3 = tensor3.storageOffset() - 1
    val stride3 = tensor3.stride(dim)
    val size3 = tensor3.size(dim)

    var isFinished = false

    while (!isFinished) {
      func(_data1, _offset1, stride1, size1,
        _data2, _offset2, stride2, size2,
        _data3, _offset3, stride3, size3)

      if (tensor1.nDimension == 1) {
        isFinished = true
      } else {
        var d = 1
        var isBreak = false
        while (d <= tensor1.nDimension && !isBreak) {
          if (d == dim) {
            if (d == tensor1.nDimension) {
              isFinished = true
              isBreak = true
            }
          } else {
            counter(d - 1) += 1
            _offset1 += tensor1.stride(d)
            _offset2 += tensor2.stride(d)
            _offset3 += tensor3.stride(d)

            if (counter(d - 1) == tensor1.size(d)) {
              if (d == tensor1.nDimension) {
                isFinished = true
                isBreak = true
              } else {
                _offset1 -= counter(d - 1) * tensor1.stride(d)
                _offset2 -= counter(d - 1) * tensor2.stride(d)
                _offset3 -= counter(d - 1) * tensor3.stride(d)
                counter(d - 1) = 0
              }
            } else {
              isBreak = true
            }
          }
          d += 1
        }
      }
    }
  }
}
