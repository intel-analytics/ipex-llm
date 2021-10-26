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

package com.intel.analytics.bigdl.ppml.vfl

import com.intel.analytics.bigdl.dllib.nn.{CAddTable, Sequential}
import com.intel.analytics.bigdl.dllib.optim.{OptimMethod, ValidationMethod, ValidationResult}

import com.intel.analytics.bigdl.dllib.utils.T
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.ppml.DLLibAggregator
import org.apache.log4j.Logger
import com.intel.analytics.bigdl.ppml.common.FLPhase._

class VflNNAggregator(classifier: Module[Float],
                      optimMethod: OptimMethod[Float],
                      criterion: Criterion[Float],
                      validationMethods: Array[ValidationMethod[Float]]) extends DLLibAggregator{
  module = Sequential[Float]().add(CAddTable[Float]())
  if (classifier != null) {
    module.add(classifier)
  }

  def setClientNum(clientNum: Int): this.type = {
    this.clientNum = clientNum
    this
  }


  override def aggregate(): Unit = {
    val inputTable = getInputTableFromStorage(TRAIN)

    val output = module.forward(inputTable)
    val loss = criterion.forward(output, target)
    val gradOutput = criterion.backward(output, target)
    val gradInput = module.backward(inputTable, gradOutput)

    // TODO: Multi gradinput
    postProcess(TRAIN, gradInput, T(loss))

  }
  // TODO: add evaluate and predict


}

object VflAggregator {
  val logger = Logger.getLogger(this.getClass)

  def apply(clientNum: Int,
            classifier: Module[Float],
            optimMethod: OptimMethod[Float],
            criterion: Criterion[Float]): VflNNAggregator = {
    new VflNNAggregator(classifier, optimMethod, criterion, null).setClientNum(clientNum)
  }

  def apply(clientNum: Int,
            classifier: Module[Float],
            optimMethod: OptimMethod[Float],
            criterion: Criterion[Float],
            validationMethods: Array[ValidationMethod[Float]]): VflNNAggregator = {
    new VflNNAggregator(classifier, optimMethod, criterion, validationMethods).setClientNum(clientNum)
  }
}