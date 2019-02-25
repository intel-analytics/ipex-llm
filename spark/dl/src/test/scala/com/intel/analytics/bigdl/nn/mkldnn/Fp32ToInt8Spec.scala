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

package com.intel.analytics.bigdl.nn.mkldnn

import java.io.File
import java.util.UUID

import com.intel.analytics.bigdl.nn.Module
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class Fp32ToInt8Spec  extends FlatSpec with Matchers with BeforeAndAfter {

  val modelPath: String = "myTestModel" + UUID.randomUUID().toString
  val weightPath: String = "myTestModelWeight" + UUID.randomUUID().toString

  "Saving and loading scale and mask" should "work properly" in {

    val myModel = Linear(3, 4)

    val assignedInputMask: Int = Random.nextInt(100)
    val assignedInputScales: Array[Array[Float]] = Array.ofDim[Float](3, 4).map(
      (arry: Array[Float]) => {
        arry.map((x: Float) => {
          Random.nextFloat()
        })
      }
    )

    val assignedOutputMask: Int = Random.nextInt()
    val assignedOutputScales: Array[Array[Float]] = Array.ofDim[Float](3, 4).map(
      (arry: Array[Float]) => {
        arry.map((x: Float) => {
          Random.nextFloat()
        })
      }
    )

    myModel.setInputDimMask(assignedInputMask)
    myModel.setInputScales(assignedInputScales)

    myModel.setOutputDimMask(assignedOutputMask)
    myModel.setOutputScales(assignedOutputScales)

    myModel.saveModule(modelPath, weightPath, true)

    val loadedModel = Module.loadModule[Float](modelPath, weightPath).asInstanceOf[Linear]

    val loadedInputMask = loadedModel.getInputDimMask()
    val loadedInputScales = loadedModel.getInputScales()
    val loadedOutputMask = loadedModel.getOutputDimMask()
    val loadedOutputScales = loadedModel.getOutputScales()

    loadedInputMask should be (assignedInputMask)
    loadedInputScales should be (assignedInputScales)

    loadedOutputMask should be (assignedOutputMask)
    loadedOutputScales should be (assignedOutputScales)

  }

  after {
    new File(modelPath).delete()
    new File(weightPath).delete()
  }
}
