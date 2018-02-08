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

class CrossCol[T: ClassTag](
  val hashBucketSize: Int,
  val strDelimiter: String = ","
)(implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[Int], T]{

  output = Tensor[Int]()

  override def updateOutput(input: Table): Tensor[Int] = {

    val tensorNum = input.length()
    require(tensorNum >=2, "the input table must contain more than one tensor")
    val batchSize = input[Tensor[String]](1).size(dim = 1)

    val indices0 = new ArrayBuffer[Int]()
    val indices1 = new ArrayBuffer[Int]()
    val values = new ArrayBuffer[Int]()
    val shape = Array(batchSize, hashBucketSize)

    (1 to tensorNum).foreach(i =>
      input[Tensor[String]](i).squeeze()
    )

    var i = 1
    while (i <= batchSize) {
      var j = 1

      val tempArr = new ArrayBuffer[Array[String]]()
      while (j <= tensorNum) {
        tempArr += input[Tensor[String]](j).valueAt(i).split(strDelimiter)
        j += 1
      }

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

    output = Tensor.sparse(
      Array(indices0.toArray, indices1.toArray),
      values.toArray,
      shape)

    output
  }

  def reCombine(input: ArrayBuffer[Array[String]]): Array[Array[String]] = {
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

  def crossHash(input: Array[Array[String]]): Array[Int] = {
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
  def apply[T](
  hashBucketSize: Int
  ) (implicit ev: TensorNumeric[T]): CrossCol[T]
  = new CrossCol[T](
    hashBucketSize = hashBucketSize
  )
}