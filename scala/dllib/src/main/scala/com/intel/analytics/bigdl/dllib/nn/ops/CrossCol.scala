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
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.hashing.MurmurHash3

/**
 * CrossCol operation preforms crosses of categorical features.
 *
 * The transformation can be thought of as Hash(cartesian product of features) % hashBucketSize)
 *
 * The input Table contains more than or equal to 2 Tensor[String].
 * Each Tensor[String] represents a categorical feature column.
 * Each row in Tensor[String] represents the string of value,
 * which supports single-value and multi-value joined by strDelimiter.
 *
 * As for the SparseTensor, it should be transformed to
 * Tensor[String] before feeding into the Operation
 *
 *
 * For example, if the two input tensors with size=3 are:
 *  "A,D", "B", "A,C"
 *  "1", "2", "3,4"
 *
 *  the output tensor should be an 2D 3 x maxLength SparseTensor:
 *  [0, 0]: Hash32("1", Hash32("D")) % hashBucketSize
 *  [0, 1]: Hash32("1", Hash32("A")) % hashBucketSize
 *  [1, 0]: Hash32("2", Hash32("B")) % hashBucketSize
 *  [2, 0]: Hash32("3", Hash32("C")) % hashBucketSize
 *  [2, 1]: Hash32("4", Hash32("C")) % hashBucketSize
 *  [2, 2]: Hash32("3", Hash32("A")) % hashBucketSize
 *  [2, 3]: Hash32("4", Hash32("A")) % hashBucketSize
 *
 * @param hashBucketSize An Int > 1. The number of buckets.
 * @param strDelimiter The Delimiter between feature values, default: ",".
 * @tparam T Numeric type. Parameter tensor numeric type. Only support float/double now
 */

class CrossCol[T: ClassTag](
  val hashBucketSize: Int,
  val strDelimiter: String = ","
)(implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[Int], T]{

  output = Tensor[Int]()

  override def updateOutput(input: Table): Tensor[Int] = {

    val tensorNum = input.length()
    require(tensorNum>=2, "the input table must contain more than one tensor")
    val batchSize = input[Tensor[String]](1).size(dim = 1)

    val indices0 = new ArrayBuffer[Int]()
    val indices1 = new ArrayBuffer[Int]()
    val values = new ArrayBuffer[Int]()

    val bufferInput = (1 to tensorNum).map { i =>
      input[Tensor[String]](i).view(input[Tensor[String]](i).size()).squeeze()
    }.toArray

    var i = 1
    var maxLen = 1
    while (i <= batchSize) {
      var j = 0
      var tempLen = 1
      val tempArr = new ArrayBuffer[Array[String]]()
      while (j < tensorNum) {
        val bufferArr = bufferInput(j).valueAt(i).split(strDelimiter)
        tempArr += bufferArr
        tempLen *= bufferArr.length
        j += 1
      }

      maxLen = if (maxLen<tempLen) tempLen else maxLen

      val resHashArr = crossHash(reCombine(tempArr))

      var m = 0
      while (m < resHashArr.length) {
        indices0 += i-1
        indices1 += m
        values += resHashArr(m)
        m += 1
      }

      i += 1
    }

    val shape = Array(batchSize, maxLen)

    output = Tensor.sparse(
      Array(indices0.toArray, indices1.toArray),
      values.toArray,
      shape)

    output
  }

  private def reCombine(input: ArrayBuffer[Array[String]]): Array[Array[String]] = {
    val stack = mutable.Stack[Array[String]]()
    stack.pushAll(input(0).map(Array(_)))

    val mkBuilder = (a: Array[String]) => mutable.ArrayBuilder.make[String]() ++= a

    val result = mutable.ArrayBuilder.make[Array[String]]()

    while (stack.nonEmpty) {
      val current = stack.pop()
      val children = input(current.length).map { nextStr =>
        val builder = mkBuilder(current) += nextStr
        builder.result()
      }
      if (current.length == input.length - 1) {
        result ++= children
      } else {
        stack.pushAll(children)
      }
    }

    result.result()
  }

  private def crossHash(input: Array[Array[String]]): Array[Int] = {
    input.map { arr =>
      var hashVal = MurmurHash3.stringHash(arr(0))
      var k = 1
      while (k < arr.length) {
        hashVal = MurmurHash3.stringHash(arr(k), hashVal)
        k += 1
      }
      hashVal % hashBucketSize match {
        case v if v < 0 => v + hashBucketSize
        case v => v
      }
    }
  }
}

object CrossCol {
  def apply[T: ClassTag](
    hashBucketSize: Int,
    strDelimiter: String = ","
  ) (implicit ev: TensorNumeric[T]): CrossCol[T]
  = new CrossCol[T](
    hashBucketSize = hashBucketSize,
    strDelimiter = strDelimiter
  )
}
