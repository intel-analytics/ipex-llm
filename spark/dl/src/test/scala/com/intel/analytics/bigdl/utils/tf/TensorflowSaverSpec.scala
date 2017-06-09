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


import java.nio.ByteOrder
import java.util.UUID

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.log4j.Logger

class TensorflowSaverSpec extends TensorflowSpecHelper {

  private val logger = Logger.getLogger(getClass)

  "ReLU layer" should "be correctly saved" in {
    val inputTensor = Tensor[Float](T(
      T(1.0f, 2.0f, 5.0f, 6.0f),
      T(-3.0f, -4.0f, -7.0f, -8.0f)
    ))
    test(ReLU[Float](), inputTensor) should be(true)
  }

  "Linear layer" should "be correctly saved" in {
    val layer = Linear[Float](3, 4,
      initWeight = Tensor(T(
        T(1.0f, 2.0f, 3.0f),
        T(4.0f, 5.0f, 6.0f),
        T(7.0f, 8.0f, 9.0f),
        T(10.0f, 11.0f, 12.0f)
      )),
      initBias = Tensor(T(1.0f, 2.0f, 3.0f, 4.0f))
    )
    val input = Tensor[Float](T(
      T(1.0f, 2.0f, 5.0f),
      T(-3.0f, -4.0f, -7.0f)
    ))
    test(layer, input, false, "/biasAdd") should be(true)
  }

  "AvgPooling" should "be correctly saved" in {
    val layer = SpatialAveragePooling(2, 2)
    val input = Tensor[Float](T(T(
      T(
        T(1.0f, 2.0f, 5.0f),
        T(-3.0f, -4.0f, -7.0f),
        T(-4.0f, -2.0f, -1.0f)
      ),
      T(
        T(-1.0f, -2.0f, -5.0f),
        T(3.0f, 4.0f, 7.0f),
        T(4.0f, 2.0f, 1.0f)
      )
    )))
    test(layer, input, true) should be(true)
  }

  "MaxPooling" should "be correctly saved" in {
    val layer = SpatialMaxPooling(2, 2)
    val input = Tensor[Float](T(T(
      T(
        T(1.0f, 2.0f, 5.0f),
        T(-3.0f, -4.0f, -7.0f),
        T(-4.0f, -2.0f, -1.0f)
      ),
      T(
        T(-1.0f, -2.0f, -5.0f),
        T(3.0f, 4.0f, 7.0f),
        T(4.0f, 2.0f, 1.0f)
      )
    )))
    test(layer, input, true) should be(true)
  }

  "Tanh" should "be correctly saved" in {
    val layer = Tanh()
    val input = Tensor[Float](4).rand()
    test(layer, input) should be(true)
  }

  "Squeeze" should "be correctly saved" in {
    val layer = Squeeze(3)
    val input = Tensor[Float](4, 2, 1, 2).rand()
    test(layer, input, true) should be(true)
  }

  private def test(layer: AbstractModule[Tensor[Float], Tensor[Float], Float],
                   inputTensor: Tensor[Float],
                   convertNHWC: Boolean = false,
                   outputSuffix: String = "") : Boolean = {
    tfCheck()
    val layerNode = layer.setName("output").apply()
    val graph = Graph(layerNode, layerNode)
    val outputTensor = layer.forward(inputTensor)

    val tmpFile = java.io.File.createTempFile("tensorflowSaverTest" + UUID.randomUUID(), "Layer")
    logger.info(s"Save model to ${tmpFile}")
    val tfTensor = if (convertNHWC) {
      inputTensor.transpose(2, 3).transpose(3, 4).contiguous()
    } else {
      inputTensor
    }
    val outputSave = if (convertNHWC) {
      outputTensor.transpose(2, 3).transpose(3, 4).contiguous()
    } else {
      outputTensor
    }
    TensorflowSaver.saveGraphWitNodeDef(
      graph,
      Seq(Tensorflow.const(tfTensor, "input", ByteOrder.LITTLE_ENDIAN)),
      tmpFile.getPath,
      ByteOrder.LITTLE_ENDIAN,
      TensorflowDataFormat.NHWC,
      Set(Tensorflow.const(outputSave, "target", ByteOrder.LITTLE_ENDIAN))
    )
    runPythonSaveTest(tmpFile.getPath, outputSuffix)
  }
}