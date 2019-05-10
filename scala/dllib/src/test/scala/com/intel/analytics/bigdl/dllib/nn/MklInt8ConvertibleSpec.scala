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

import java.util.UUID
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class MklInt8ConvertibleSpec extends FlatSpec with Matchers with BeforeAndAfter  {


  "Unit test setInputDimMask" should "work properly" in {
    val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1")
    val conv2 = SpatialConvolution(1, 6, 5, 5).setName("conv2")
    val seq = Sequential().add(conv1).add(conv2)

    // Input mask default to 0
    seq.getInputDimMask() should be (0)
    conv1.getInputDimMask() should be (0)
    conv2.getInputDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag off
    // the submodules conv1 & conv2 should not be effected
    seq.setInputDimMask(1, false)
    seq.getInputDimMask() should be (1)
    conv1.getInputDimMask() should be (0)
    conv2.getInputDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag on
    // the submodules conv1 & conv2 should be effected
    seq.setInputDimMask(2, true)
    seq.getInputDimMask() should be (2)
    conv1.getInputDimMask() should be (2)
    conv2.getInputDimMask() should be (2)

    // change conv1's input dimension mask
    conv1.setInputDimMask(4, false)
    seq.getInputDimMask() should be (2)
    conv1.getInputDimMask() should be (4)
    conv2.getInputDimMask() should be (2)

    // change conv2's input dimension mask
    conv2.setInputDimMask(8, false)
    seq.getInputDimMask() should be (2)
    conv1.getInputDimMask() should be (4)
    conv2.getInputDimMask() should be (8)

  }


  "Unit test setOutputDimMask" should "work properly" in {
    val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1")
    val conv2 = SpatialConvolution(1, 6, 5, 5).setName("conv2")
    val seq = Sequential().add(conv1).add(conv2)

    // Input mask default to 0
    seq.getOutputDimMask() should be (0)
    conv1.getOutputDimMask() should be (0)
    conv2.getOutputDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag off
    // the submodules conv1 & conv2 should not be effected
    seq.setOutputDimMask(1, false)
    seq.getOutputDimMask() should be (1)
    conv1.getOutputDimMask() should be (0)
    conv2.getOutputDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag on
    // the submodules conv1 & conv2 should be effected
    seq.setOutputDimMask(2, true)
    seq.getOutputDimMask() should be (2)
    conv1.getOutputDimMask() should be (2)
    conv2.getOutputDimMask() should be (2)

    // change conv1's input dimension mask
    conv1.setOutputDimMask(4, false)
    seq.getOutputDimMask() should be (2)
    conv1.getOutputDimMask() should be (4)
    conv2.getOutputDimMask() should be (2)

    // change conv2's input dimension mask
    conv2.setOutputDimMask(8, false)
    seq.getOutputDimMask() should be (2)
    conv1.getOutputDimMask() should be (4)
    conv2.getOutputDimMask() should be (8)

  }

  "Unit test setWeightDimMask" should "work properly" in {
    val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1")
    val conv2 = SpatialConvolution(1, 6, 5, 5).setName("conv2")
    val seq = Sequential().add(conv1).add(conv2)

    // Input mask default to 0
    seq.getWeightDimMask() should be (0)
    conv1.getWeightDimMask() should be (0)
    conv2.getWeightDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag off
    // the submodules conv1 & conv2 should not be effected
    seq.setWeightDimMask(1, false)
    seq.getWeightDimMask() should be (1)
    conv1.getWeightDimMask() should be (0)
    conv2.getWeightDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag on
    // the submodules conv1 & conv2 should be effected
    seq.setWeightDimMask(2, true)
    seq.getWeightDimMask() should be (2)
    conv1.getWeightDimMask() should be (2)
    conv2.getWeightDimMask() should be (2)

    // change conv1's input dimension mask
    conv1.setWeightDimMask(4, false)
    seq.getWeightDimMask() should be (2)
    conv1.getWeightDimMask() should be (4)
    conv2.getWeightDimMask() should be (2)

    // change conv2's input dimension mask
    conv2.setWeightDimMask(8, false)
    seq.getWeightDimMask() should be (2)
    conv1.getWeightDimMask() should be (4)
    conv2.getWeightDimMask() should be (8)

  }

}

