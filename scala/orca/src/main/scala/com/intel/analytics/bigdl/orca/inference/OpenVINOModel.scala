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

import java.util.{ArrayList, Arrays, List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import scala.collection.JavaConverters._

class OpenVINOModel(var executableNetworkReference: Long = -1,
                    var supportive: OpenVinoInferenceSupportive,
                    var isInt8: Boolean = false)
  extends AbstractModel with InferenceSupportive with Serializable {

  override def predict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    val outputs = new ArrayList[JList[JTensor]]()
    inputs.asScala.map(input => {
      val tensor = input.get(0)
      val output = if (isInt8) {
        supportive.predictInt8(executableNetworkReference,
          tensor.getData, tensor.getShape)
      } else {
        supportive.predict(executableNetworkReference,
          tensor.getData, tensor.getShape)
      }
      outputs.add(Arrays.asList({
        output
      }))
    })
    outputs
  }

  override def predict(inputActivity: Activity): Activity = {
    val (inputList, batchSize) = inputActivity.isTable match {
      case true =>
        val inputTable = inputActivity.toTable
        val batchSize = inputTable.length()
        (transferBatchTableToJListOfJListOfJTensor(inputTable, batchSize), batchSize)
      case false =>
        val inputTensor = inputActivity.toTensor[Float]
        val batchSize = inputTensor.size(1)
        (transferBatchTensorToJListOfJListOfJTensor(inputTensor, batchSize), batchSize)
    }
    val outputs = predict(inputList)
    transferListOfActivityToActivityOfBatch(outputs, batchSize)
  }

  override def copy(num: Int): Array[AbstractModel] = Array(this)

  override def release(): Unit = {
    isReleased match {
      case true =>
      case false =>
        supportive.releaseOpenVINOIR(executableNetworkReference)
        executableNetworkReference = -1
    }
  }

  override def isReleased(): Boolean = {
    executableNetworkReference == -1
  }

  override def toString: String = s"OpenVinoInferenceModel with " +
    s"executableNetworkReference: $executableNetworkReference, supportive: $supportive"
}
