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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._

import scala.reflect.ClassTag

class Cast[T: ClassTag, @specialized(Int, Double, Long, Float) DataType: ClassTag]()
  (implicit ev: TensorNumeric[T], ev2: TensorNumeric[DataType])
  extends Operation[Tensor[_], Tensor[DataType], T] {

  output = Activity.allocate[Tensor[DataType], DataType]()

  override def updateOutput(input: Tensor[_]): Tensor[DataType] = {
    output.resizeAs(input)
    input.getType() match {
      case FloatType =>
        output.getType() match {
          case FloatType => output.applyFun[Float](
            input.toTensor[Float],
            x => x.asInstanceOf[DataType])
          case DoubleType => output.applyFun[Float](
            input.toTensor[Float],
            x => x.toDouble.asInstanceOf[DataType])
          case IntType => output.applyFun[Float](
            input.toTensor[Float],
            x => x.toInt.asInstanceOf[DataType])
          case ShortType => output.applyFun[Float](
            input.toTensor[Float],
            x => x.toShort.asInstanceOf[DataType])
          case LongType => output.applyFun[Float](
            input.toTensor[Float],
            x => x.toLong.asInstanceOf[DataType])
          case _ => throw new RuntimeException("Unsupported tensor type")
        }
      case DoubleType =>
        output.getType() match {
          case FloatType => output.applyFun[Double](
            input.toTensor[Double],
            x => x.asInstanceOf[DataType])
          case DoubleType => output.applyFun[Double](
            input.toTensor[Double],
            x => x.asInstanceOf[DataType])
          case IntType => output.applyFun[Double](
            input.toTensor[Double],
            x => x.toInt.asInstanceOf[DataType])
          case ShortType => output.applyFun[Double](
            input.toTensor[Double],
            x => x.toShort.asInstanceOf[DataType])
          case LongType => output.applyFun[Double](
            input.toTensor[Double],
            x => x.toLong.asInstanceOf[DataType])
          case _ => throw new RuntimeException("Unsupported tensor type")
        }
      case LongType =>
        output.getType() match {
          case FloatType => output.applyFun[Long](
            input.toTensor[Long],
            x => x.asInstanceOf[DataType])
          case DoubleType => output.applyFun[Long](
            input.toTensor[Long],
            x => x.toDouble.asInstanceOf[DataType])
          case IntType => output.applyFun[Long](
            input.toTensor[Long],
            x => x.toInt.asInstanceOf[DataType])
          case ShortType => output.applyFun[Long](
            input.toTensor[Long],
            x => x.toShort.asInstanceOf[DataType])
          case LongType => output.applyFun[Long](
            input.toTensor[Long],
            x => x.asInstanceOf[DataType])
          case _ => throw new RuntimeException("Unsupported tensor type")
        }
      case ShortType =>
        output.getType() match {
          case FloatType => output.applyFun[Short](
            input.toTensor[Short],
            x => x.asInstanceOf[DataType])
          case DoubleType => output.applyFun[Short](
            input.toTensor[Short],
            x => x.toDouble.asInstanceOf[DataType])
          case IntType => output.applyFun[Short](
            input.toTensor[Short],
            x => x.toInt.asInstanceOf[DataType])
          case ShortType => output.applyFun[Short](
            input.toTensor[Short],
            x => x.asInstanceOf[DataType])
          case LongType => output.applyFun[Short](
            input.toTensor[Short],
            x => x.toLong.asInstanceOf[DataType])
          case _ => throw new RuntimeException("Unsupported tensor type")
        }
      case IntType =>
        output.getType() match {
          case FloatType => output.applyFun[Int](
            input.toTensor[Int],
            x => x.asInstanceOf[DataType])
          case DoubleType => output.applyFun[Int](
            input.toTensor[Int],
            x => x.toDouble.asInstanceOf[DataType])
          case IntType => output.applyFun[Int](
            input.toTensor[Int],
            x => x.asInstanceOf[DataType])
          case ShortType => output.applyFun[Int](
            input.toTensor[Int],
            x => x.toShort.asInstanceOf[DataType])
          case LongType => output.applyFun[Int](
            input.toTensor[Int],
            x => x.toLong.asInstanceOf[DataType])
          case _ => throw new RuntimeException("Unsupported tensor type")
        }
      case _ => throw new RuntimeException("Unsupported tensor type")
    }

    output
  }
}

object Cast {
  def apply[T: ClassTag, DataType: ClassTag]()
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[DataType]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new Cast[T, DataType]())
}
