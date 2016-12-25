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

package com.intel.analytics.bigdl.dataset.text

import com.intel.analytics.bigdl.dataset.{MiniBatch, Transformer}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.collection.Iterator

object TensorSeqToBatch {
  def apply(batchSize: Int = 1): TensorSeqToBatch = new TensorSeqToBatch(batchSize)
}

class TensorSeqToBatch(batchSize: Int = 1)
  extends Transformer[(Tensor[Float], Tensor[Float]), MiniBatch[Float]] {

  override def apply(prev: Iterator[(Tensor[Float], Tensor[Float])]): Iterator[MiniBatch[Float]] = {
    new Iterator[MiniBatch[Float]] {

      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[Float] = {
        if (prev.hasNext) {
          val data = prev.next()
          MiniBatch(data._1, data._2)
        } else {
          null
        }
      }
    }
  }
}
