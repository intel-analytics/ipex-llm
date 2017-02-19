/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.fasterrcnn.model

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.models.fasterrcnn.layers._
import com.intel.analytics.bigdl.models.fasterrcnn.model.Phase._


class VggFRcnn(phase: PhaseType = TEST) extends FasterRcnn(phase) {

  def createVgg16(): Sequential[Float] = {
    val vggNet = Sequential()
    def addConvRelu(param: (Int, Int, Int, Int, Int), name: String, isBack: Boolean = true)
    : Unit = {
      vggNet.add(conv(param, s"conv$name", isBack))
      vggNet.add(ReLU(true).setName(s"relu$name"))
    }
    addConvRelu((3, 64, 3, 1, 1), "1_1", isBack = false)
    addConvRelu((64, 64, 3, 1, 1), "1_2", isBack = false)
    vggNet.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool1"))

    addConvRelu((64, 128, 3, 1, 1), "2_1", isBack = false)
    addConvRelu((128, 128, 3, 1, 1), "2_2", isBack = false)
    vggNet.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool2"))

    addConvRelu((128, 256, 3, 1, 1), "3_1")
    addConvRelu((256, 256, 3, 1, 1), "3_2")
    addConvRelu((256, 256, 3, 1, 1), "3_3")
    vggNet.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool3"))

    addConvRelu((256, 512, 3, 1, 1), "4_1")
    addConvRelu((512, 512, 3, 1, 1), "4_2")
    addConvRelu((512, 512, 3, 1, 1), "4_3")
    vggNet.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool4"))

    addConvRelu((512, 512, 3, 1, 1), "5_1")
    addConvRelu((512, 512, 3, 1, 1), "5_2")
    addConvRelu((512, 512, 3, 1, 1), "5_3")
    vggNet
  }

  def createRpn(): Sequential[Float] = {
    val rpnModel = Sequential()
    rpnModel.add(conv((512, 512, 3, 1, 1), "rpn_conv/3x3", init = (0.01, 0.0)))
    rpnModel.add(ReLU(true).setName("rpn_relu/3x3"))
    val clsAndReg = ConcatTable()
    val clsSeq = Sequential()
    clsSeq.add(conv((512, 18, 1, 1, 0), "rpn_cls_score", init = (0.01, 0.0)))
    clsSeq.add(ReshapeInfer(Array(0, 2, -1, 0)))
    phase match {
      case TEST =>
        clsSeq.add(SoftMax())
          .add(ReshapeInfer(Array(1, 2 * param.anchorParam.num, -1, 0)))
      case _ =>
    }
    clsAndReg.add(clsSeq)
      .add(conv((512, 36, 1, 1, 0), "rpn_bbox_pred", init = (0.01, 0.0)))
    rpnModel.add(clsAndReg)
    rpnModel
  }

  def createFeatureAndRpnNet(): Sequential[Float] = {
    val compose = Sequential()
    compose.add(createVgg16())
    val vggRpnModel = ConcatTable()
    vggRpnModel.add(createRpn())
    vggRpnModel.add(Identity())
    compose.add(vggRpnModel)
    compose
  }

  protected def createFastRcnn(): Sequential[Float] = {
    val model = Sequential()
      .add(RoiPooling(pool, pool, 0.0625f).setName("pool5"))
      .add(ReshapeInfer(Array(-1, 512 * pool * pool)))
      .add(linear((512 * pool * pool, 4096), "fc6"))
      .add(ReLU())
      .add(Dropout().setName("drop6"))
      .add(linear((4096, 4096), "fc7"))
      .add(ReLU())
      .add(Dropout().setName("drop7"))

    val cls = Sequential().add(linear((4096, 21), "cls_score", (0.01, 0.0)))
    if (isTest) cls.add(SoftMax().setName("cls_prob"))
    val clsReg = ConcatTable()
      .add(cls)
      .add(linear((4096, 84), "bbox_pred", (0.001, 0.0)))

    model.add(clsReg)
    model
  }

  override val pool: Int = 7
  override val param: FasterRcnnParam = new VggParam(phase)

  override def criterion4: ParallelCriterion[Float] = {
    val rpn_loss_bbox = SmoothL1CriterionWithWeights(3.0)
    val rpn_loss_cls = SoftmaxWithCriterion(ignoreLabel = Some(-1))
    val loss_bbox = SmoothL1CriterionWithWeights(1.0)
    val loss_cls = SoftmaxWithCriterion()
    val pc = ParallelCriterion()
    pc.add(rpn_loss_cls, 1.0f)
    pc.add(rpn_loss_bbox, 1.0f)
    pc.add(loss_cls, 1.0f)
    pc.add(loss_bbox, 1.0f)
    pc
  }
}

