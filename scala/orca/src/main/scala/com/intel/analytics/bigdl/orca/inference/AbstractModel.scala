/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.pipeline.inference

import java.util.{List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.Activity

trait AbstractModel {
  def predict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]]
  def predict(inputActivity: Activity): Activity
  def copy(num: Int): Array[AbstractModel]
  def release(): Unit
  def isReleased(): Boolean
}
