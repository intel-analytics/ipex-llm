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

package com.intel.analytics.zoo.tfpark

import com.intel.analytics.zoo.common.Utils
import org.tensorflow.DataType

class TFTrainingHelperV2(graphRunner: GraphRunner,
                         checkpointPath: String,
                         inputs: Array[String],
                         inputTypes: Array[Int],
                         additionalInputs: Array[String],
                         additionalInputTypes: Array[Int],
                         labels: Array[String],
                         labelTypes: Array[Int],
                         outputs: Array[String],
                         metrics: Array[String],
                         variables: Array[String],
                         variableTypes: Array[Int],
                         variableAssignPlaceholders: Array[String],
                         assignVariableOp: String,
                         extraVariables: Array[String],
                         extraVariableTypes: Array[Int],
                         extraVariableAssignPlaceholders: Array[String],
                         assignExtraVariableOP: String,
                         gradVariables: Array[String],
                         updateOp: String,
                         private val trainOp: String,
                         initOp: Option[String],
                         defaultTensorValue: Array[Array[Float]])
  extends TFTrainingHelper(graphRunner, checkpointPath, inputs, inputTypes,
    additionalInputs, additionalInputTypes, labels, labelTypes, outputs,
    metrics, variables, variableTypes, variableAssignPlaceholders, assignVariableOp,
    extraVariables, extraVariableTypes, extraVariableAssignPlaceholders, assignExtraVariableOP,
    gradVariables, updateOp, initOp, defaultTensorValue) {

  @transient
  private var shouldUpdateParameter = false


  override def beforeRunGradient(): Unit = {

    if (!weightsRestored) {
      Utils.timeIt("setTrainingVariableIntoTF") {
        setVariableIntoTF(weights, variableAssignPlaceholders,
          variableTypes.map(TFUtils.tfenum2datatype), assignVariableOp)
      }
      weightsRestored = true
    }

    if (shouldUpdateParameter) {
      graphRunner.runTargets(targets = Vector(trainOp),
        inputs = weights.toVector, inputNames = gradVariables.toVector,
        inputTypes = Vector.fill(gradVariables.length)(DataType.FLOAT))
      shouldUpdateParameter = false
    }


    if (!extraParameterRestored) {
      setVariableIntoTF(extraParameters, extraVariableAssignPlaceholders,
        extraVariableTypes.map(TFUtils.tfenum2datatype), assignExtraVariableOP)
      extraParameterRestored = true
    }
  }

  override def afterRunGradient(): Unit = {
    super.afterRunGradient()
    shouldUpdateParameter = true
  }

  def moveWeightsOutOfTF(): Unit = {
    if (shouldUpdateParameter) {
      graphRunner.runTargets(targets = Vector(trainOp),
        inputs = weights.toVector, inputNames = gradVariables.toVector,
        inputTypes = Vector.fill(gradVariables.length)(DataType.FLOAT))
      shouldUpdateParameter = false
    }
    getVariableFromTF(weights, variableNames = variables)
    if (extraParameters.length > 0) {
      Utils.timeIt("getExtraVariableFromTF") {
        getVariableFromTF(extraParameters, variableNames = extraVariables)
      }
    }
  }

}


