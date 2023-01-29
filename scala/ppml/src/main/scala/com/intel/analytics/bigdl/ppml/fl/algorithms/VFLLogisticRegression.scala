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

package com.intel.analytics.bigdl.ppml.fl.algorithms

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.nn.{Linear, Sequential}
import com.intel.analytics.bigdl.dllib.optim.Adam
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.fl.NNModel
import com.intel.analytics.bigdl.ppml.fl.nn.VFLNNEstimator
import com.intel.analytics.bigdl.ppml.fl.utils.FLClientClosable

/**
 * VFL Logistic Regression
 * @param featureNum
 * @param learningRate
 */
class VFLLogisticRegression(featureNum: Int = -1,
                            learningRate: Float = 0.005f,
                            customModel: Module[Float] = null,
                            algorithm: String = "vfl_logistic_regression") extends NNModel() {
  Log4Error.invalidInputError(featureNum != -1 || customModel != null,
    "Either featureNum or customModel should be provided")
  val clientModule = if (customModel == null) {
    Linear[Float](featureNum, 1)
  } else customModel
  val model = Sequential[Float]().add(clientModule)
  override val estimator = new VFLNNEstimator(
    algorithm, model, new Adam(learningRate))

}
