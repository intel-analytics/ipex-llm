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

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.{Activities, T, Table}

import scala.reflect.ClassTag

/**
 * ParallelCriterion is a weighted sum of other criterions each applied to a different input
 * and target. Set repeatTarget = true to share the target for criterions.
 *
 * Use add(criterion[, weight]) method to add criterion. Where weight is a scalar(default 1).
 *
 * @param repeatTarget Whether to share the target for all criterions.
 */
class ParallelCriterion[T: ClassTag](val repeatTarget: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends Criterion[Table, T] {

  // list of sub criterions
  val criterions = T()
  val weights = T()
  var gradInput = T()

  def add(criterion: Criterion[_ <: Activities, T], weight : Double = 1.0): this.type = {
    criterions.insert(criterion)
    weights.insert(weight)
    this
  }

  override def updateOutput(input: Table, target: Table): T = {
    var output = ev.fromType[Int](0)
    var i = 1
    while(i <= criterions.length()) {
      val currentCriterion = criterions[Criterion[Activities, T]](i)
      val currentTarget: Activities = if (repeatTarget) target else target(i)
      output = ev.plus(output, ev.times(weights[T](i),
        currentCriterion.forward(input(i), currentTarget))
      )
      i += 1
    }

    output
  }

  override def updateGradInput(input: Table, target: Table): Table = {
    gradInput = Utils.recursiveResizeAs[T](gradInput, input).toTable()
    Utils.recursiveFill[T](gradInput, 0)
    var i = 1
    while (i <= criterions.length()) {
      val currentCriterion = criterions[Criterion[Activities, T]](i)
      val currentTarget: Activities = if (repeatTarget) target else target(i)
      Utils.recursiveAdd[T](gradInput(i), weights(i),
        currentCriterion.updateGradInput(input(i), currentTarget))
      i += 1
    }

    gradInput
  }
}
