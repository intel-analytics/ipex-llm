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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{CAddTable, CDivTable, CMulTable, CSubTable}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * [[Operation]] is an abstract class which represents Tensorflow's operations
 *
 * @tparam A Input data type
 * @tparam T Numeric type. Only support float/double now
 */
abstract class Operation[A <: Activity: ClassTag,
@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T])
  extends AbstractModule[A, Tensor[T], T]{

  override def updateGradInput(input: A, gradOutput: Tensor[T]): A = {
    throw new UnsupportedOperationException("Operation does not support updateGradInput() method")
  }

  override def accGradParameters(input: A, gradOutput: Tensor[T]): Unit = {
    throw new UnsupportedOperationException("Operation does not support updateGradInput() method")
  }

  override def backward(input: A, gradOutput: Tensor[T]): A = {
    throw new UnsupportedOperationException("Operation does not support backward() method")
  }
}

/**
 * Wrap nn models as an Tensorflow operation, if an nn module's function
 * exactly corresponds to an Tensoflow operation.
 *
 * @param module an nn module
 * @tparam A Input data type
 * @tparam T Numeric type. Only support float/double now
 */
class OperationWrapper[A <: Activity: ClassTag, T: ClassTag]
(module: AbstractModule[A, Tensor[T], T])
  (implicit ev: TensorNumeric[T])
  extends Operation[A, T]{

  override def updateOutput(input: A): Tensor[T] = {
    output = module.forward(input)
    output
  }
}

object OperationWrapper {
  def apply[A <: Activity: ClassTag, T: ClassTag](model: AbstractModule[A, Tensor[T], T])
    (implicit ev: TensorNumeric[T]): OperationWrapper[A, T] = new OperationWrapper(model)
}

object Add {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Table, T]
  = OperationWrapper[Table, T](CAddTable())
}

object Subtract {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Table, T]
  = OperationWrapper[Table, T](CSubTable())
}

object Multiply {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Table, T]
   = OperationWrapper[Table, T](CMulTable())
}

object Divide {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Table, T]
  = OperationWrapper[Table, T](CDivTable())
}

