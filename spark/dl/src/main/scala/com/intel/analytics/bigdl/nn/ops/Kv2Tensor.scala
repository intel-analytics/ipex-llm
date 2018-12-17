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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Kv2Tensor operation convert a kv feature column to a SparseTensor or DenseTensor
 *
 * DenseTensor if transType = 0
 * SparseTensor if  transType = 1
 *
 * The input contains 2 elements which are `kvTensor`, `feaLen`:
 * kvTensor shape will be batch*1 and element is a kv string, only support one feature now
 * depth: the length of the value set of the feature
 *
 * the output shape will be batch*feaLen if transType = 0
 * the output shape will be a SparseTensor with dense shape batch*feaLen if transType = 1
 *
 * @param kvDelimiter The delimiter between kv pairs, default: ","
 * @param itemDelimiter The delimiter between key and value, default: ":"
 * @param transType The type of output tensor. default: 0
 * @tparam T Numeric type. Parameter tensor numeric type. Only support float/double now
 * @tparam D Numeric type. Output tensor numeric type. Only support float/double now
 */

class Kv2Tensor[T: ClassTag, D: ClassTag](
  val kvDelimiter: String,
  val itemDelimiter: String,
  val transType: Int
  )(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T]{

  output = Activity.allocate[Tensor[D], D]()

  override def updateOutput(input: Table): Tensor[D] = {
    val kvTensor = input[Tensor[String]](1)
    val feaLen = input[Tensor[Int]](2).value()
    val indices0 = new ArrayBuffer[Int]()
    val indices1 = new ArrayBuffer[Int]()
    val values = new ArrayBuffer[D]()
    val rows = kvTensor.size(dim = 1)
    val shape = Array(rows, feaLen)

    var i = 1
    while(i<=rows) {
      val kvFeaString = kvTensor.select(1, i).valueAt(1)
      kvFeaString.split(kvDelimiter).foreach { kv =>
        indices0 += i-1
        indices1 += kv.split(itemDelimiter)(0).toInt
        ev2.getType() match {
          case DoubleType =>
            values += kv.split(itemDelimiter)(1).toDouble.asInstanceOf[D]
          case FloatType =>
            values += kv.split(itemDelimiter)(1).toFloat.asInstanceOf[D]
          case t => throw new NotImplementedError(s"$t is not supported")
        }
      }
      i += 1
    }

    val indices = Array(indices0.toArray, indices1.toArray)
    val resTensor = transType match {
      case 0 =>
        Tensor.dense(Tensor.sparse(indices, values.toArray, shape))
      case 1 =>
        Tensor.sparse(indices, values.toArray, shape)
    }
    output = resTensor
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object Kv2Tensor{
  def apply[T: ClassTag, D: ClassTag](
     kvDelimiter: String = ",",
     itemDelimiter: String = ":",
     transType: Int = 0)
     (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): Kv2Tensor[T, D]
  = new Kv2Tensor[T, D](
    kvDelimiter = kvDelimiter,
    itemDelimiter = itemDelimiter,
    transType = transType
  )
}

