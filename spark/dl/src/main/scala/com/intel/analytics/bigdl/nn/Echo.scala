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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleData, ModuleSerializable, SerializeContext}
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule

import scala.reflect.ClassTag

/**
 * This module is for debug purpose, which can print activation and gradient in your model
 * topology
 *
 * User can pass in a customized function to inspect more information from the activation. This is
 * very useful in Debug.
 *
 * Please note that the passed in customized function will not be persisted in serialization.
 */
@SerialVersionUID(6735245897546687343L)
class Echo[T: ClassTag](
  private var feval: (Echo[T], Tensor[T]) => Unit,
  private var beval: (Echo[T], Tensor[T], Tensor[T]) => Unit
) (implicit ev: TensorNumeric[T])
  extends TensorModule[T]  {

  /**
   * Set evaluation method for forward
   * @param feval
   * @return
   */
  def setFeval(feval: (Echo[T], Tensor[T]) => Unit): this.type = {
    this.feval = feval
    this
  }

  /**
   * Set evaluation method for backward
   * @param beval
   * @return
   */
  def setBeval(beval: (Echo[T], Tensor[T], Tensor[T]) => Unit): this.type = {
    this.beval = beval
    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    this.output = input
    feval(this, input)
    this.output
  }
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput = gradOutput
    beval(this, input, gradOutput)
    this.gradInput
  }
}

object Echo extends ModuleSerializable {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]) : Echo[T] = {
    new Echo[T](Echo.defaultFeval[T]_, Echo.defaultBeval[T]_)
  }

  def apply[T: ClassTag](feval: (Echo[T], Tensor[T]) => Unit)
    (implicit ev: TensorNumeric[T]) : Echo[T] = {
    new Echo[T](feval, Echo.defaultBeval[T]_)
  }

  def apply[T: ClassTag](feval: (Echo[T], Tensor[T]) => Unit,
    beval: (Echo[T], Tensor[T], Tensor[T]) => Unit)
    (implicit ev: TensorNumeric[T]) : Echo[T] = {
    new Echo[T](feval, beval)
  }

  private def defaultFeval[T](module: Echo[T], input: Tensor[T]): Unit = {
    println(s"${module.getPrintName} : Activation size is ${input.size().mkString("x")}")
  }

  private def defaultBeval[T](module: Echo[T], input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    println(s"${module.getPrintName} : Gradient size is ${gradOutput.size().mkString("x")}")
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              b: BigDLModule.Builder)
                                                  (implicit ev: TensorNumeric[T]): Unit = {
    // We won't serialize the function, so do nothing here
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
      (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    new Echo[T](defaultFeval, defaultBeval)
  }
}
