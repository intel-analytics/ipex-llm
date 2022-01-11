/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.base

import com.intel.analytics.bigdl.dllib.feature.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

trait Estimator {
  protected val evaluateResults: mutable.Map[String, ArrayBuffer[Float]]
  def getEvaluateResults(): Map[String, Array[Float]] = {
    evaluateResults.map(v => (v._1, v._2.toArray)).toMap
  }
  def train(endEpoch: Int,
            trainDataSet: LocalDataSet[MiniBatch[Float]],
            valDataSet: LocalDataSet[MiniBatch[Float]]): Any

  def evaluate(dataSet: LocalDataSet[MiniBatch[Float]])

  def predict(dataSet: LocalDataSet[MiniBatch[Float]]): Array[Activity]
}
