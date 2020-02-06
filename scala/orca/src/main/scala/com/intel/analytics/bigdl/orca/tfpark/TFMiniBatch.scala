/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.tfpark

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, Transformer, Utils}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{Tensor, TensorNumericMath}
import com.intel.analytics.bigdl.utils.T


class TFMiniBatch(data: Array[Tensor[_]]) extends MiniBatch[Float] {
  override def size(): Int = data(0).size(1)

  override def slice(offset: Int, length: Int): MiniBatch[Float] = {
    val newData = new Array[Tensor[_]](data.length)
    var i = 0
    while (i < data.length) {
      newData(i) = data(i).narrow(1, offset, length)
      i += 1
    }
    new TFMiniBatch(newData)
  }

  override def getInput(): Activity = {
    if (data.length == 1) {
      data(0)
    } else {
      val t = T()
      var i = 0
      while (i < data.length) {
        t.insert(data(i))
        i += 1
      }
      t
    }
  }

  override def getTarget(): Activity = {
    null // fake target, should new be used
  }

  override def set(samples: Seq[Sample[Float]])
                  (implicit ev: TensorNumericMath.TensorNumeric[Float]): TFMiniBatch.this.type = {
    throw new UnsupportedOperationException("TFMiniBatch does not support set method")
  }
}

object TFMiniBatch {
  def apply(data: Array[Tensor[_]]): TFMiniBatch = new TFMiniBatch(data)
}

class StringToMiniBatch(batchSize: Int, partitionNum: Option[Int] = None)
  extends Transformer[Array[Byte], TFMiniBatch] {
  import TFTensorNumeric.NumericByteArray

  private val batchPerPartition = Utils.getBatchSize(batchSize, partitionNum)

  override def apply(prev: Iterator[Array[Byte]]): Iterator[TFMiniBatch] = {
    new Iterator[TFMiniBatch] {

      override def hasNext: Boolean = prev.hasNext

      override def next(): TFMiniBatch = {
        val sampleData = new Array[Array[Byte]](batchPerPartition)
        if (prev.hasNext) {
          var i = 0
          while (i < batchPerPartition && prev.hasNext) {
            val sample = prev.next()
            sampleData(i) = sample
            i += 1
          }

          if (i < batchPerPartition) {
            val tensor = Tensor[Array[Byte]](sampleData.slice(0, i), shape = Array[Int](i))
            TFMiniBatch(Array(tensor))
          } else {
            val tensor = Tensor[Array[Byte]](sampleData, shape = Array[Int](batchPerPartition))
            TFMiniBatch(Array(tensor))
          }
        } else {
          null
        }
      }
    }
  }
}
