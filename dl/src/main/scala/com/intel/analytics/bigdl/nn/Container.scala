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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Activities, Table}
import com.intel.analytics.bigdl._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

abstract class Container[A <: Activities : ClassTag,
    B <: Activities : ClassTag, T: ClassTag](
  implicit ev: TensorNumeric[T]) extends Module[A, B, T] {

  // list of sub modules
  val modules: ArrayBuffer[Module[Activities, Activities, T]]
  = ArrayBuffer[Module[Activities, Activities, T]]()

  def add(module: Module[_ <: Activities, _ <: Activities, T]): this.type = {
    modules += module.asInstanceOf[Module[Activities, Activities, T]]
    this
  }

  override def zeroGradParameters(): Unit = {
    modules.foreach(_.zeroGradParameters())
  }

  override def updateParameters(learningRate: T): Unit = {
    modules.foreach(_.updateParameters(learningRate))
  }

  override def reset(): Unit = {
    modules.foreach(_.reset())
  }

  override def training(): this.type = {
    train = true
    modules.foreach(_.training())
    this
  }

  override def evaluate(): this.type = {
    train = false
    modules.foreach(_.evaluate())
    this
  }


//  /**
//    * Find a module by given a parameter offset
//    *
//    * @param paramOffset parameter offset in the (weight, grad) vector returned by the
//    *                    getParamter function
//    * @param indexes     ignore it
//    * @return module ref, offset(ignore), indexes from the current module
//    */
//  override def findModel(
//                 paramOffset: Int,
//                 indexes: Array[Int] = Array()):
//  (Module[_ <: Activities, _ <: Activities, T], Int, Array[Int]) = (this, paramOffset, indexes)
//
//  override def mapModules(f: Module[_ <: Activities, _ <: Activities, T] => Unit): Unit = {
//    f(this)
//
//    if (modules != null) {
//      modules.foreach(_.mapModules(f))
//    }
//  }
//
//  override def findModules(name: String): ArrayBuffer[Module[_ <: Activities, _ <: Activities, T]] = {
//    def matchName(module: Module[_ <: Activities, _ <: Activities, T]) =
//      module.getClass.getName.equals(name)
//
//    val nodes = new ArrayBuffer[Module[_ <:Activities, _ <:Activities, T]]()
//
//    if (matchName(this)) nodes.append(this)
//    if (modules != null) {
//      modules
//        .foreach(m => {
//          if (matchName(m)) nodes.append(m)
//          else if (m.isInstanceOf[Container[_ <: Activities, _ <: Activities, T]]) {
//            val tempNodes = m.findModules(name)
//            nodes ++= tempNodes
//          }
//      })
//    }
//
//    nodes
//  }


  override def getTimes():
    Array[(Module[_ <: Activities, _ <: Activities, T], Long, Long)] = {
    this.modules.flatMap(_.getTimes()).toArray
  }

  override def resetTimes(): Unit = {
    modules.foreach(_.resetTimes())
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    val weights = new ArrayBuffer[Tensor[T]]()
    val gradWeights = new ArrayBuffer[Tensor[T]]()
    modules.foreach(m => {
      val params = m.parameters()
      if (params != null) {
        params._1.foreach(weights += _)
        params._2.foreach(gradWeights += _)
      }
    })
    (weights.toArray, gradWeights.toArray)
  }

  override def clearState() : this.type = {
    super.clearState()
    modules.foreach(_.clearState())
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Container[A, B, T]]

  override def equals(other: Any): Boolean = other match {
    case that: Container[A, B, T] =>
      super.equals(that) &&
        (that canEqual this) &&
        modules == that.modules
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), modules)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}
