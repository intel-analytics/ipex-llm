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

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.mkldnn
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, ReflectionUtils}
import com.intel.analytics.bigdl.utils.intermediate.{IRToBlas, IRToDnn}
import com.intel.analytics.bigdl.{Module, nn}

class ReflectionUtilsSpec extends BigDLSpecHelper {

  "test SpatialConvolution reflection" should "be right" in {
    val model1 = nn.SpatialConvolution[Float](2, 4, 3, 3, 4, 4, 0, 0).asInstanceOf[Module[Float]]
    val className = "com.intel.analytics.bigdl.utils.intermediate.IRSpatialConvolution"
    val cls = Class.forName(className)
    val ir = ReflectionUtils.reflectToIR[Float](model1, cls)
    val cls2 = Class.forName(
      "com.intel.analytics.bigdl.nn.SpatialConvolution")
    val modelBlas = ReflectionUtils.reflectFromIR(ir, cls2)

    val cls3 = Class.forName("com.intel.analytics.bigdl.nn.mkldnn.SpatialConvolution")
    val modelDnn = ReflectionUtils.reflectFromIR(ir, cls3).asInstanceOf[mkldnn.SpatialConvolution]

    val inputShape = Array(2, 2, 23, 23)
    val outShape = Array(2, 4, 6, 6)

    val seq = Sequential()
        .add(Input(inputShape, Memory.Format.nchw))
        .add(modelDnn)
        .add(Output(Memory.Format.nchw))
    seq.compile(TrainingPhase)

    val input = Tensor[Float](inputShape).rand()
    val gradOutput = Tensor[Float](outShape).rand()

    val out = model1.forward(input).toTensor[Float]
    val out1 = modelBlas.forward(input).toTensor[Float]
    val out2 = seq.forward(input).toTensor[Float]

    out should be(out1)
    Equivalent.nearequals(out1, Tools.dense(out2).toTensor[Float], 1e-4) should be(true)

    val grad = model1.backward(input, gradOutput)
    val grad1 = modelBlas.backward(input, gradOutput)
    val grad2 = seq.backward(input, gradOutput)

    val gradWeight1 = modelDnn.getParameters()._2
    val gradWeight2 = modelBlas.getParameters()._2

    val weight1 = modelDnn.getParameters()._1
    val weight2 = modelBlas.getParameters()._1

    Equivalent.nearequals(weight1, weight2) should be (true)
    Equivalent.nearequals(gradWeight1, gradWeight2) should be (true)

    Equivalent.nearequals(Tools.dense(seq.gradInput).toTensor,
      modelBlas.gradInput.toTensor[Float]) should be (true)
  }

  "test BatchNorm reflection" should "be right" in {
    val model1 = nn.SpatialBatchNormalization(3).asInstanceOf[Module[Float]]
    val className = "com.intel.analytics.bigdl.utils.intermediate.IRSpatialBatchNormalization"
    val cls = Class.forName(className)
    val ir = ReflectionUtils.reflectToIR[Float](model1, cls)

    val modelBlas = IRToBlas[Float].convertLayer(ir)
    val modelDnn = IRToDnn[Float].convertLayer(ir).asInstanceOf[mkldnn.SpatialBatchNormalization]

    val inputShape = Array(16, 3, 4, 4)
    modelDnn.setRuntime(new MklDnnRuntime)
    modelDnn.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    modelDnn.initBwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    modelDnn.initGradWPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)


    val input = Tensor[Float](16, 3, 4, 4).rand()
    val gradOutput = Tensor[Float](16, 3, 4, 4).rand()

    val out1 = modelBlas.forward(input).toTensor[Float]
    val out2 = modelDnn.forward(input).toTensor[Float]

    Equivalent.nearequals(out1, Tools.dense(out2).toTensor[Float], 1e-4) should be(true)

    val grad1 = modelBlas.backward(input, gradOutput)
    val grad2 = modelDnn.backward(input, gradOutput)

    val gradWeight1 = Tools.dense(modelDnn.gradWeightAndBias.native).toTensor
    val gradWeight2 = modelBlas.getParameters()._2

    val weight1 = Tools.dense(modelDnn.weightAndBias.native).toTensor
    val weight2 = modelBlas.getParameters()._1

    Equivalent.nearequals(weight1, weight2) should be (true)
    Equivalent.nearequals(gradWeight1, gradWeight2, 1e-4) should be (true)

    Equivalent.nearequals(Tools.dense(modelDnn.gradInput).toTensor,
      modelBlas.gradInput.toTensor[Float]) should be (true)
  }
}
