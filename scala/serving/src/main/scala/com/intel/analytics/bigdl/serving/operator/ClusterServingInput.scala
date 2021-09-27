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

package com.intel.analytics.bigdl.serving.operator

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T

class ClusterServingInput(name: String, input: Array[(Array[Float], Array[Int])]) {
  def getActivity(): Activity = {
    if (input.size == 0) {
      Tensor[Float]()
    }
    else if (input.size == 1) {
      Tensor[Float](input(0)._1, input(0)._2)
    } else {
      T(input.head, input.tail: _*)
    }
  }
  def getName(): String = this.name
}
object ClusterServingInput {
  /**
   * To construct a single element input
   * @param name input name
   * @param input (value, shape) tuple
   * @return
   */
  def apply(name: String, input: (Array[Float], Array[Int])): Activity = {
    new ClusterServingInput(name, Array(input)).getActivity()
  }

  def apply(name: String, input: Array[(Array[Float], Array[Int])]): Activity = {
    new ClusterServingInput(name, input).getActivity()
  }

  def apply(name: String, input: Array[Float]): Activity = {
    new ClusterServingInput(name, Array((input, Array(input.size)))).getActivity()
  }

  def apply(name: String, input: Array[String]): Activity = {
    val t = Tensor[String](input.size)
    (0 until input.size).foreach(i => t.setValue(i, input(i)))
    t
  }
}
