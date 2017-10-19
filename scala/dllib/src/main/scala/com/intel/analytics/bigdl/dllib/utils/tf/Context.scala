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
package com.intel.analytics.bigdl.utils.tf

import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable

class Context[T](
  tensorsMap: mutable.HashMap[String, (Tensor[T], Tensor[T], Option[Seq[(Int, Int)]])]
  ) {

  /**
   * Return weight, gradient and shape info
   * @param key
   * @return
   */
  def apply(key: String): (Tensor[T], Tensor[T], Option[Seq[(Int, Int)]]) = tensorsMap(key)

  def update(key: String, value: (Tensor[T], Tensor[T], Option[Seq[(Int, Int)]])): this.type = {
    tensorsMap(key) = value
    this
  }

  def containsTensor(key: String): Boolean = tensorsMap.contains(key)

  def tensors(): Iterable[(Tensor[T], Tensor[T], Option[scala.Seq[(Int, Int)]])] = {
    tensorsMap.values
  }

  def putTensor(key: String, value: (Tensor[T], Tensor[T], Option[Seq[(Int, Int)]])): this.type = {
    update(key, value)
  }

  def tensorNames(): Set[String] = {
    tensorsMap.keySet.toSet
  }

  def this() = this(new mutable.HashMap[String, (Tensor[T], Tensor[T], Option[Seq[(Int, Int)]])]())

  def assignGrads : Option[Set[(String, String)]] = assignGradsAndWeights

  def setAssignGrads(set: Set[(String, String)]): this.type = {
    assignGradsAndWeights = Some(set)
    this
  }

  private var assignGradsAndWeights : Option[Set[(String, String)]] = None
}
