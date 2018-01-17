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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.{Sequential => TSequential}
import com.intel.analytics.bigdl.nn.{DynamicGraph, Graph, StaticGraph}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.ops._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Edge, Node}
import com.intel.analytics.bigdl.utils.serializer._
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.runtime._

object Model {
  /**
   * Build multiple inputs, multiple outputs graph container.
   * @param input input node
   * @param output output node
   * @return a graph container
   */
  def apply[T: ClassTag](
                          input : Array[ModuleNode[T]],
                          output : Array[ModuleNode[T]],
                          variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None
                        )(implicit ev: TensorNumeric[T]) : Graph[T] = {
    new StaticGraph[T](input, output, variables)
  }

  /**
   * Build a single input, multiple outputs graph container
   * @param input input node
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : Array[ModuleNode[T]])
                        (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new StaticGraph[T](Seq(input), output)
  }

  /**
   * Build a multiple inputs, single output graph container
   * @param input input nodes
   * @param output output node
   * @return a graph container
   */
  def apply[T: ClassTag](input : Array[ModuleNode[T]], output : ModuleNode[T])
                        (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new StaticGraph[T](input, Seq(output))
  }
  /**
   * Build a single input, single output graph container
   * @param input input nodes
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : ModuleNode[T])
                        (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new StaticGraph[T](Seq(input), Seq(output))
  }
}

class Sequential[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TSequential[T] {

  /**
   * Add a sub-module to the contained `modules`
   *
   * @param module module to be add
   * @return this container
   */
  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    this.excludeTorch[T](Seq(Node(module)))
    if (this.modules.isEmpty) {
      if (module.getInputShape() == null) {
        throw new RuntimeException("The first layer should explicitly declare inputshape")
      } else {
        module.build(module.getInputShape())
      }
    } else {
      module.build(this.getOutputShape())
    }
    modules += module.asInstanceOf[AbstractModule[Activity, Activity, T]]
    this
  }
}

object Sequential {
  def apply[@specialized(Float, Double) T: ClassTag]()
     (implicit ev: TensorNumeric[T]) : Sequential[T] = {
    new Sequential[T]()
  }
}
