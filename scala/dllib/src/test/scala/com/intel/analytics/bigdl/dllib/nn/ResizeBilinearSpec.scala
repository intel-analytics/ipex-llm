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
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class ResizeBilinearSpec extends FlatSpec with Matchers {
  private val input = Tensor[Float](T(T(
    T(
      T(1, 2, 3),
      T(4, 5, 6)
    ),
    T(
      T(7, 8, 9),
      T(2, 3, 1)
    ),
    T(
      T(4, 8, 2),
      T(5, 3, 0)
    )
  )))

  "ResizeBilinear forward" should "not change content while input/output width/height match" in {
    println(input)
    val layer = ResizeBilinear[Float](3, 2, dataFormat = DataFormat.NHWC)
    val output = layer.forward(input)
    println(output)
    input should be(output)
  }

  "ResizeBilinear forward" should "be correct while double height" in {
    println(input)
    val layer = ResizeBilinear[Float](6, 2, dataFormat = DataFormat.NHWC)
    val output = layer.forward(input)
    println(output)
    val expectOutput = Tensor[Float](T(T(
      T(
        T(1, 2, 3),
        T(4, 5, 6)
      ),
      T(
        T(4, 5, 6),
        T(3, 4, 3.5)
      ),
      T(
        T(7, 8, 9),
        T(2, 3, 1)
      ),
      T(
        T(5.5, 8, 5.5),
        T(3.5, 3, 0.5)
      ),
      T(
        T(4, 8, 2),
        T(5, 3, 0)
      ),
      T(
        T(4, 8, 2),
        T(5, 3, 0)
      )
    )))
    output should be(expectOutput)
  }

  "ResizeBilinear forward" should "be correct while double width" in {
    println(input)
    val layer = ResizeBilinear[Float](3, 4, dataFormat = DataFormat.NHWC)
    val output = layer.forward(input)
    println(output)
    val expectOutput = Tensor[Float](T(T(
      T(
        T(1, 2, 3),
        T(2.5, 3.5, 4.5),
        T(4, 5, 6),
        T(4, 5, 6)
      ),
      T(
        T(7, 8, 9),
        T(4.5, 5.5, 5),
        T(2, 3, 1),
        T(2, 3, 1)
      ),
      T(
        T(4, 8, 2),
        T(4.5, 5.5, 1),
        T(5, 3, 0),
        T(5, 3, 0)
      )
    )))
    output should be(expectOutput)
  }

  "ResizeBilinear forward and backward" should "be correct with NCHW" in {

    case class Param(inHeight: Int, inWidth: Int,
                     outHeight: Int, outWidth: Int, alignCorners: Boolean)
    val params = Seq(
      Param(3, 2, 3, 2, true),
      Param(3, 2, 6, 2, true),
      Param(3, 2, 3, 4, true),
      Param(3, 2, 3, 2, false),
      Param(3, 2, 6, 2, false),
      Param(3, 2, 3, 4, false)
    )

    for (param <- params) {
      val inputCFirst = Tensor[Float](Array(1, 3, param.inHeight, param.inWidth)).rand()
      val inputCLast = inputCFirst.clone().transpose(2, 4).transpose(2, 3).contiguous()

      val gradOutputCFirst = Tensor[Float](Array(1, 3, param.outHeight, param.outWidth)).rand()
      val gradOutputCLast = gradOutputCFirst.clone().transpose(2, 4).transpose(2, 3).contiguous()
      val layerCLast = ResizeBilinear[Float](param.outHeight, param.outWidth,
        param.alignCorners, dataFormat = DataFormat.NHWC)
      val layerCFirst = ResizeBilinear[Float](param.outHeight, param.outWidth,
        param.alignCorners, dataFormat = DataFormat.NCHW)

      // NCHW
      val outputCFirst = layerCFirst.forward(inputCFirst)
      val gradInputCFirst = layerCFirst.backward(inputCFirst, gradOutputCFirst)

      val outputCLast = layerCLast.forward(inputCLast)
      val gradInputCLast = layerCLast.backward(inputCLast, gradOutputCLast)

      outputCFirst
        .transpose(2, 4)
        .transpose(2, 3).contiguous() should be(outputCLast)

      gradInputCFirst
        .transpose(2, 4)
        .transpose(2, 3).contiguous() should be(gradInputCLast)
    }
  }
}

class ResizeBilinearSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val input = Tensor[Float](1, 3, 2, 3).apply1( _ => Random.nextFloat())
    val resizeBilinear = ResizeBilinear[Float](3, 2).setName("resizeBilinear")
    runSerializationTest(resizeBilinear, input)
  }
}
