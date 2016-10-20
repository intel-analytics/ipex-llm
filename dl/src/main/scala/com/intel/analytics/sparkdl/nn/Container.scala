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

package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.utils.Table
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.{Activities, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

private[nn] abstract class Container[A <: Activities : ClassTag, B <: Activities : ClassTag, @specialized(Float, Double) T: ClassTag](
  implicit ev: TensorNumeric[T]) extends Module[A, B, T] {

  def add(module: Module[_ <: Activities, _ <: Activities, T]): this.type = {
    modules += module
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
    modules.foreach(_.training())
    this
  }

  override def evaluate(): this.type = {
    modules.foreach(_.evaluate())
    this
  }

  override def getTimes(): Array[(Module[_ <: Activities, _ <: Activities, T], Long, Long)] = {
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

  override def findModel(paramOffset: Int,
    indexes: Array[Int]): (Module[_ <: Activities, _ <: Activities, T], Int, Array[Int]) = {
    var offset = paramOffset
    var result: Module[_ <: Activities, _ <: Activities, T] = this.asInstanceOf[Module[Activities, Activities, T]]
    var newIndexes = indexes
    var i = 0
    modules.foreach(m => {
      if (result == this) {
        val r = m.findModel(offset, indexes ++ Array(i))
        if (r._2 <= 0) {
          result = r._1
          newIndexes = r._3
        }
        offset = r._2
        i += 1
      }
    })
    (result, offset, newIndexes)
  }
}
