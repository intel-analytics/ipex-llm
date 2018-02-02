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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.hashing.MurmurHash3

/**
 * CategoricalColHashBucket operation can convert feature string to a Sparse/Dense Tensor
 *
 * SparseTensor if isSparse = true
 * DenseTensor if  isSparse = false
 *
 * the input is a Tensor[String] with shape batch * 1.
 *
 * This operation distributes your inputs into a finite number of buckets by hashing
 *
 * The Operation support the feature column with single-value or multi-value
 *
 * The output_id = Hash(input_feature_string) % hashBucketSize, ranging 0 to hashBucketSize-1
 *
 * The missing values in input Tensor can be represented by -1 for int and '''' for string
 *
 * @param hashBucketSize An Integer > 1. The number of buckets.
 * @param strDelimiter The delimiter of feature string, default: ",".
 * @param isSparse whether the output tensor is a sparseTensor, default: True.
 * @tparam T Numeric type. Parameter tensor numeric type. Only support float/double now
 */

class CategoricalColHashBucket[T: ClassTag](
  val hashBucketSize: Int,
  val strDelimiter: String = ",",
  val isSparse: Boolean = true
  )(implicit ev: TensorNumeric[T])
  extends Operation[Tensor[String], Tensor[Int], T] {

  output = Tensor[Int]()

  override def updateOutput(input: Tensor[String]): Tensor[Int] = {
    val rows = input.size(dim = 1)
    val indices0 = new ArrayBuffer[Int]()
    val indices1 = new ArrayBuffer[Int]()
    val values = new ArrayBuffer[Int]()
    var i = 1
    var max_fea_len = 0
    while(i <= rows) {
      val feaStrArr = input.valueAt(i, 1).split(strDelimiter)
      max_fea_len = math.max(max_fea_len, feaStrArr.length)
      var j = 0
      while(j < feaStrArr.length) {
        val hashVal = MurmurHash3.stringHash(feaStrArr(j)) % hashBucketSize match {
          case v if v < 0 => v + hashBucketSize
          case v => v
        }
        indices0 += i-1
        indices1 += j
        values += hashVal
        j += 1
      }
      i += 1
    }
    val indices = Array(indices0.toArray, indices1.toArray)
    val shape = Array(rows, max_fea_len)
    output = isSparse match {
      case true =>
        Tensor.sparse(indices, values.toArray, shape)
      case false =>
        Tensor.dense(Tensor.sparse(indices, values.toArray, shape))
    }
    output
  }
}

object CategoricalColHashBucket{
  def apply[T: ClassTag](
      hashBucketSize: Int,
      strDelimiter: String = ",",
      isSparse: Boolean = true)
      (implicit ev: TensorNumeric[T])
  : CategoricalColHashBucket[T] = new CategoricalColHashBucket[T](
    hashBucketSize = hashBucketSize,
    strDelimiter = strDelimiter,
    isSparse = isSparse
  )
}
