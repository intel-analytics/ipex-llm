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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, Shape, TestUtils}

import scala.util.Random

class SpatialSeparableConvolutionSpec extends BigDLSpecHelper {
  "SpatialSeparableConvolution NHWC and NCHW" should "have same output" in {
    val depthWeightNHWC = Tensor[Float](2, 2, 3, 1).rand()
    val depthWeightNCHW = depthWeightNHWC.transpose(1, 4).transpose(2, 4).transpose(2, 3)
      .contiguous()
    val pointWeightNHWC = Tensor[Float](1, 1, 3, 6).rand()
    val pointWeightNCHW = pointWeightNHWC.transpose(1, 4).transpose(2, 4).transpose(2, 3)
      .contiguous()
    val convNHWC = SpatialSeparableConvolution[Float](3, 6, 1, 2, 2, dataFormat = DataFormat.NHWC,
      initDepthWeight = depthWeightNHWC, initPointWeight = pointWeightNHWC)
    val convNCHW = SpatialSeparableConvolution[Float](3, 6, 1, 2, 2, dataFormat = DataFormat.NCHW,
      initDepthWeight = depthWeightNCHW, initPointWeight = pointWeightNCHW)
    val inputNHWC = Tensor[Float](2, 24, 24, 3).rand()
    val inputNCHW = inputNHWC.transpose(2, 4).transpose(3, 4).contiguous()
    val outputNHWC = convNHWC.forward(inputNHWC)
    val outputNCHW = convNCHW.forward(inputNCHW)
    val convert = outputNHWC.transpose(2, 4).transpose(3, 4).contiguous()
    convert.almostEqual(outputNCHW, 1e-5) should be(true)
    val gradOutputNHWC = Tensor[Float](2, 23, 23, 6).rand()
    val gradOutputNCHW = gradOutputNHWC.transpose(2, 4).transpose(3, 4).contiguous()
    val gradInputNHWC = convNHWC.backward(inputNHWC, gradOutputNHWC)
    val gradInputNCHW = convNCHW.backward(inputNCHW, gradOutputNCHW)
    val convertGradInput = gradInputNHWC.transpose(2, 4).transpose(3, 4).contiguous()
    convertGradInput.almostEqual(gradInputNCHW, 1e-5) should be(true)

    convNHWC.parameters()._2.zip(convNCHW.parameters()._2).map { case(p1, p2) =>
      if (p1.nDimension() == 4) {
        val convert = p2.transpose(1, 4).transpose(1, 3).transpose(2, 3)
        p1.almostEqual(convert, 1e-3) should be(true)
      } else {
        p1.almostEqual(p2, 1e-3) should be(true)
      }
    }
  }

  "SpatialSeparableConvolution NHWC and NCHW" should "have same output when depth mul is 2" in {
    val depthWeightNHWC = Tensor[Float](2, 2, 3, 2).rand()
    val depthWeightNCHW = depthWeightNHWC.transpose(1, 4).transpose(2, 4).transpose(2, 3)
      .contiguous()
    val pointWeightNHWC = Tensor[Float](1, 1, 6, 6).rand()
    val pointWeightNCHW = pointWeightNHWC.transpose(1, 4).transpose(2, 4).transpose(2, 3)
      .contiguous()
    val convNHWC = SpatialSeparableConvolution[Float](3, 6, 2, 2, 2, dataFormat = DataFormat.NHWC,
      initDepthWeight = depthWeightNHWC, initPointWeight = pointWeightNHWC)
    val convNCHW = SpatialSeparableConvolution[Float](3, 6, 2, 2, 2, dataFormat = DataFormat.NCHW,
      initDepthWeight = depthWeightNCHW, initPointWeight = pointWeightNCHW)
    val inputNHWC = Tensor[Float](2, 24, 24, 3).rand()
    val inputNCHW = inputNHWC.transpose(2, 4).transpose(3, 4).contiguous()
    val outputNHWC = convNHWC.forward(inputNHWC)
    val outputNCHW = convNCHW.forward(inputNCHW)
    val convert = outputNHWC.transpose(2, 4).transpose(3, 4).contiguous()
    convert.almostEqual(outputNCHW, 1e-5) should be(true)
    val gradOutputNHWC = Tensor[Float](2, 23, 23, 6).rand()
    val gradOutputNCHW = gradOutputNHWC.transpose(2, 4).transpose(3, 4).contiguous()
    val gradInputNHWC = convNHWC.backward(inputNHWC, gradOutputNHWC)
    val gradInputNCHW = convNCHW.backward(inputNCHW, gradOutputNCHW)
    val convertGradInput = gradInputNHWC.transpose(2, 4).transpose(3, 4).contiguous()
    convertGradInput.almostEqual(gradInputNCHW, 1e-5) should be(true)

    convNHWC.parameters()._2.zip(convNCHW.parameters()._2).map { case(p1, p2) =>
      if (p1.nDimension() == 4) {
        val convert = p2.transpose(1, 4).transpose(1, 3).transpose(2, 3)
        p1.almostEqual(convert, 1e-3) should be(true)
      } else {
        p1.almostEqual(p2, 1e-3) should be(true)
      }
    }
  }

  "SpatialSeparableConvolution" should "be able to serialized" in {
    val conv = SpatialSeparableConvolution[Float](3, 6, 2, 2, 2)
    val file = createTmpFile()
    conv.saveModule(file.getAbsolutePath, overWrite = true)
    val conv2 = Module.loadModule[Float](file.getAbsolutePath)
  }

  "SpatialSeparableConvolution computeOutputShape NCHW" should "work properly" in {
    val layer = SpatialSeparableConvolution[Float](3, 6, 1, 2, 2)
    TestUtils.compareOutputShape(layer, Shape(3, 12, 12)) should be (true)
  }

  "SpatialSeparableConvolution computeOutputShape NHWC" should "work properly" in {
    val layer = SpatialSeparableConvolution[Float](2, 5, 2, 2, 1, dataFormat = DataFormat.NHWC)
    TestUtils.compareOutputShape(layer, Shape(24, 24, 2)) should be (true)
  }

}

class SpatialSeparableConvolutionSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val separableConv = SpatialSeparableConvolution[Float](2, 2, 1, 2, 2,
      dataFormat = DataFormat.NHWC).setName("separableConv")
    val input = Tensor[Float](1, 5, 5, 2).apply1( e => Random.nextFloat())
    runSerializationTest(separableConv, input)
  }
}
