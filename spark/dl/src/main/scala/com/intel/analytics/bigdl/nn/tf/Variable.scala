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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.ops.Operation
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag


class Variable[T: ClassTag](val variableValue: Tensor[T], val variableGradient: Tensor[T])
                           (implicit ev: TensorNumeric[T])
  extends Operation[Activity, Tensor[T], T] with WithoutInput{

  override def clearState(): this.type = {
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.variableValue), Array(this.variableGradient))
  }

  override def updateOutput(input: Activity): Tensor[T] = {
    this.output.resizeAs(variableValue)
    this.output.copy(variableValue)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = {
    require(gradOutput.isSameSizeAs(variableValue),
      s"Invalid gradOutput size. require (${variableValue.size().mkString(",")}), but " +
        s"(${gradOutput.size().mkString(",")})")
    input match {
      case t: Tensor[T] =>
        if (gradInput == null || gradInput.isInstanceOf[Table]) {
          gradInput = Tensor[T]()
        }
        gradInput.toTensor[T].resizeAs(t).zero()
      case t: Table =>
        if (gradInput == null || !gradInput.isInstanceOf[Table]) {
          gradInput = T()
        }
        t.foreach(kv => {
          val gradInputTensors = gradInput.toTable
          val grad = gradInputTensors.getOrElse[Tensor[T]](kv._1, Tensor[T]())
            .resizeAs(kv._2.asInstanceOf[Tensor[T]]).zero()
          gradInputTensors(kv._1) = grad
        })
    }
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Tensor[T]): Unit = {
    this.variableGradient.add(ev.fromType[Double](1.0), gradOutput)
  }
}

object Variable {
  def apply[T: ClassTag](variableValue: Tensor[T], variableGradient: Tensor[T])
    (implicit ev: TensorNumeric[T]): Variable[T] = new Variable(variableValue, variableGradient)
}
