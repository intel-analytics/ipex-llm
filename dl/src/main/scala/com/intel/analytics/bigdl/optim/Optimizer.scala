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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{File, Table}

import scala.collection.mutable.ArrayBuffer

abstract class Optimizer[@specialized(Float, Double) T](
  protected val model: Module[Tensor[T], Tensor[T], T],
  protected val endWhen: Trigger
) {
  protected var validationTrigger: Option[Trigger] = None
  protected var cacheTrigger: Option[Trigger] = None
  protected val validationMethods: ArrayBuffer[ValidationMethod[T]] = new ArrayBuffer()
  protected var cachePath: Option[String] = None
  protected var isOverWrite: Boolean = false

  def optimize(): Module[Tensor[T], Tensor[T], T]

  def setValidationTrigger(trigger: Trigger): this.type = {
    this.validationTrigger = Some(trigger)
    this
  }

  def addValidation(validationMethod: ValidationMethod[T]): this.type = {
    validationMethods.append(validationMethod)
    this
  }

  def setCache(path: String, trigger: Trigger): this.type = {
    this.cachePath = Some(path)
    this.cacheTrigger = Some(trigger)
    this
  }

  def overWriteCache() : this.type = {
    isOverWrite = true
    this
  }

  protected def saveModel(postfix: String = ""): this.type = {
    if (this.cachePath.isDefined) {
      model.save(s"${cachePath.get}.model$postfix", isOverWrite)
    }
    this
  }

  protected def saveState(state: Table, postfix: String = ""): this.type = {
    if (this.cachePath.isDefined) {
      state.save(s"${cachePath.get}.state$postfix", isOverWrite)
    }
    this
  }
}

trait Trigger {
  def apply(state: Table): Boolean
}

object Trigger {
  def everyEpoch: Trigger = {
    new Trigger() {
      private var lastEpoch = -1

      override def apply(state: Table): Boolean = {
        if (lastEpoch == -1) {
          lastEpoch = state[Int]("epoch")
          false
        } else {
          if (state[Int]("epoch") == lastEpoch) {
            false
          } else {
            lastEpoch = state[Int]("epoch")
            true
          }
        }
      }
    }
  }

  def severalIteration(interval: Int): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        val curIteration = state[Int]("neval")
        curIteration != 0 && curIteration % interval == 0
      }
    }
  }

  def maxEpoch(max: Int): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        state[Int]("epoch") > max
      }
    }
  }

  def maxIteration(max: Int): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        state[Int]("neval") > max
      }
    }
  }
}
