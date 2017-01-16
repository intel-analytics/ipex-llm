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

import com.intel.analytics.bigdl.models.fasterrcnn.layers._
import com.intel.analytics.bigdl.models.fasterrcnn.model.Phase._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor


class PvanetFRcnn(phase: PhaseType = TEST)
  extends FasterRcnn(phase) {

  var pvanet: Sequential[Float] = _

  private def concatNeg(name: String): Concat[Float] = {
    val concat = Concat(2)
    concat.add(Identity())
    concat.add(Power(1, -1, 0).setName(s"$name/neg"))
    concat.setName(s"$name/concat")
    concat
  }


  private def addScaleRelu(module: Sequential[Float],
    sizes: Array[Int], name: String): Unit = {
    module.add(Scale(sizes).setName(name))
    module.add(ReLU())
  }

  private def addConvComponent(compId: Int, index: Int, p: Array[(Int, Int, Int, Int, Int)]) = {
    val label = s"${compId}_$index"
    val convTable = ConcatTable()
    val conv_left = Sequential()
    var i = 0
    if (index == 1) {
      conv_left.add(conv(p(i), s"conv$label/1/conv"))
      i += 1
    } else {
      conv_left.add(conv(p(i), s"conv$label/1/conv"))
      i += 1
    }

    conv_left.add(ReLU())
    conv_left.add(conv(p(i), s"conv$label/2/conv"))
    i += 1
    conv_left.add(concatNeg(s"conv$label/2"))
    if (compId == 2) {
      addScaleRelu(conv_left, Array(1, 48, 1, 1), s"conv$label/2/scale")
    } else {
      addScaleRelu(conv_left, Array(1, 96, 1, 1), s"conv$label/2/scale")
    }

    conv_left.add(conv(p(i), s"conv$label/3/conv"))
    i += 1

    convTable.add(conv_left)
    if (index == 1) {
      convTable.add(conv(p(i), s"conv$label/proj"))
      i += 1
    } else {
      convTable.add(Identity())
    }
    pvanet.add(convTable)
    pvanet.add(CAddTable().setName(s"conv$label"))
  }

  private def addInception(module: Sequential[Float], label: String, index: Int,
    p: Array[(Int, Int, Int, Int, Int)]): Unit = {
    val left = Sequential()
    val incep = Concat(2)

    var i = 0
    val com1 = Sequential()
    com1.add(conv(p(i), s"conv$label/incep/0/conv")).add(ReLU())
    i += 1
    incep.add(com1)

    val com2 = Sequential()
    com2.add(conv(p(i), s"conv$label/incep/1_reduce/conv")).add(ReLU())
    i += 1
    com2.add(conv(p(i), s"conv$label/incep/1_0/conv")).add(ReLU())
    i += 1
    incep.add(com2)

    val com3 = Sequential()
    com3.add(conv(p(i), s"conv$label/incep/2_reduce/conv")).add(ReLU())
    i += 1
    com3.add(conv(p(i), s"conv$label/incep/2_0/conv")).add(ReLU())
    i += 1
    com3.add(conv(p(i), s"conv$label/incep/2_1/conv")).add(ReLU())
    i += 1
    incep.add(com3)

    if (index == 1) {
      val com4 = Sequential()
      com4.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName(s"conv$label/incep/pool"))
      com4.add(conv(p(i), s"conv$label/incep/poolproj/conv")).add(ReLU())
      i += 1
      incep.add(com4)
    }

    left.add(incep)
    left.add(conv(p(i), s"conv$label/out/conv"))
    i += 1
    val table = ConcatTable()
    table.add(left)
    if (index == 1) {
      table.add(conv(p(i), s"conv$label/proj"))
      i += 1
    } else {
      table.add(Identity())
    }
    module.add(table)
    module.add(CAddTable().setName(s"conv$label"))
  }


  private def getPvanet: Sequential[Float] = {
    if (pvanet != null) return pvanet
    pvanet = Sequential()
    pvanet.add(conv((3, 16, 7, 2, 3), "conv1_1/conv"))

    pvanet.add(concatNeg("conv1_1"))
    addScaleRelu(pvanet, Array(1, 32, 1, 1), "conv1_1/scale")
    pvanet.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1"))


    addConvComponent(2, 1, Array((32, 24, 1, 1, 0), (24, 24, 3, 1, 1),
      (48, 64, 1, 1, 0), (32, 64, 1, 1, 0)))
    var i = 2
    while (i <= 3) {
      addConvComponent(2, i, Array((64, 24, 1, 1, 0), (24, 24, 3, 1, 1), (48, 64, 1, 1, 0)))
      i += 1
    }

    addConvComponent(3, 1, Array((64, 48, 1, 2, 0), (48, 48, 3, 1, 1),
      (96, 128, 1, 1, 0), (64, 128, 1, 2, 0)))

    i = 2
    while (i <= 4) {
      addConvComponent(3, i, Array((128, 48, 1, 1, 0), (48, 48, 3, 1, 1), (96, 128, 1, 1, 0)))
      i += 1
    }

    val inceptions4_5 = Sequential()

    val inceptions4 = Sequential()
    addInception(inceptions4, "4_1", 1, Array((128, 64, 1, 2, 0), (128, 48, 1, 2, 0),
      (48, 128, 3, 1, 1), (128, 24, 1, 2, 0), (24, 48, 3, 1, 1), (48, 48, 3, 1, 1),
      (128, 128, 1, 1, 0), (368, 256, 1, 1, 0), (128, 256, 1, 2, 0)))

    i = 2
    while (i <= 4) {
      addInception(inceptions4, s"4_$i", i, Array((256, 64, 1, 1, 0), (256, 64, 1, 1, 0),
        (64, 128, 3, 1, 1), (256, 24, 1, 1, 0), (24, 48, 3, 1, 1),
        (48, 48, 3, 1, 1), (240, 256, 1, 1, 0)))
      i += 1
    }
    inceptions4_5.add(inceptions4)


    val seq5 = Sequential()
    val inceptions5 = Sequential()
    addInception(inceptions5, "5_1", 1, Array((256, 64, 1, 2, 0), (256, 96, 1, 2, 0),
      (96, 192, 3, 1, 1), (256, 32, 1, 2, 0), (32, 64, 3, 1, 1), (64, 64, 3, 1, 1),
      (256, 128, 1, 1, 0), (448, 384, 1, 1, 0), (256, 384, 1, 2, 0)))
    i = 2
    while (i <= 4) {
      addInception(inceptions5, s"5_$i", i, Array((384, 64, 1, 1, 0), (384, 96, 1, 1, 0),
        (96, 192, 3, 1, 1), (384, 32, 1, 1, 0), (32, 64, 3, 1, 1), (64, 64, 3, 1, 1),
        (320, 384, 1, 1, 0)))
      i += 1
    }

    seq5.add(inceptions5)
    seq5.add(new SpatialFullConvolution[Tensor[Float], Float](384, 384, 4, 4, 2, 2, 1, 1,
      nGroup = 384, noBias = true).setName("upsample"))

    val concat5 = Concat(2)
    concat5.add(Identity())
    concat5.add(seq5)

    inceptions4_5.add(concat5)

    val concatConvf = Concat(2).setName("concat")
    concatConvf.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("downsample"))
    concatConvf.add(inceptions4_5)
    pvanet.add(concatConvf)

    pvanet
  }

  override def createFeatureAndRpnNet(): Sequential[Float] = {
    val compose = Sequential()
    compose.add(getPvanet)

    val convTable = new ConcatTable[Float]
    convTable.add(Sequential()
      .add(conv((768, 128, 1, 1, 0), "convf_rpn"))
      .add(ReLU()))
    convTable.add(Sequential()
      .add(conv((768, 384, 1, 1, 0), "convf_2"))
      .add(ReLU()))
    compose.add(convTable)
    val rpnAndFeature = ConcatTable()
    rpnAndFeature.add(Sequential()
      .add(new SelectTable(1)).add(createRpn()))
    rpnAndFeature.add(JoinTable(2, 4))
    compose.add(rpnAndFeature)
    compose
  }

  protected def createFastRcnn(): Sequential[Float] = {
    val model = Sequential()
      .add(RoiPooling(pool, pool, 0.0625f).setName("roi_pool_conv5"))
      .add(ReshapeInfer(Array(-1, 512 * pool * pool)).setName("roi_pool_conv5_reshape"))
    if (isTrain) {
      model.add(linear((512 * pool * pool, 4096), "fc6_"))
      model.add(BatchNormalization(4096).setName("fc6/bn"))
      model.add(Scale(Array(4096)).setName("fc6/scale"))
      model.add(Dropout().setName("fc6/dropout"))
    } else {
      model.add(linear((512 * pool * pool, 4096), "fc6"))
      model.add(ReLU())
    }
    model.add(linear((4096, 4096), "fc7"))
    if (isTrain) {
      model.add(BatchNormalization(4096).setName("fc7/bn"))
      model.add(Scale(Array(4096)).setName("fc7/scale"))
      model.add(Dropout().setName("fc7/dropout"))
    } else {
      model.add(ReLU())
    }

    val cls = Sequential().add(linear((4096, 21), "cls_score"))
    if (isTest) cls.add(SoftMax())
    val clsReg = ConcatTable()
      .add(cls)
      .add(linear((4096, 84), "bbox_pred"))

    model.add(clsReg)
    model
  }

  def createRpn(): Sequential[Float] = {
    val rpnModel = Sequential()
    rpnModel.add(conv((128, 384, 3, 1, 1), "rpn_conv1"))
    rpnModel.add(ReLU())
    val clsAndReg = ConcatTable()
    val clsSeq = Sequential()
    clsSeq.add(conv((384, 50, 1, 1, 0), "rpn_cls_score", init = (0.01, 0.0)))
    clsSeq.add(ReshapeInfer(Array(0, 2, -1, 0)).setName("rpn_cls_score_reshape"))
    phase match {
      case TEST =>
        clsSeq.add(SoftMax())
          .add(ReshapeInfer(Array(1, 2 * param.anchorParam.num, -1, 0))
            .setName("rpn_cls_prob_reshape"))
      case _ =>
    }
    clsAndReg.add(clsSeq).add(conv((384, 100, 1, 1, 0), "rpn_bbox_pred"))
    rpnModel.add(clsAndReg)
    rpnModel
  }

  override val pool: Int = 6
  override val param: FasterRcnnParam = new PvanetParam(phase)

  override def criterion4: ParallelCriterion[Float] = {
    val rpn_loss_bbox = SmoothL1CriterionWithWeights(3.0)
    val rpn_loss_cls = SoftmaxWithCriterion(ignoreLabel = Some(-1))
    val loss_bbox = SmoothL1CriterionWithWeights(1.0)
    val loss_cls = SoftmaxWithCriterion(ignoreLabel = Some(-1))
    val pc = ParallelCriterion()
    pc.add(rpn_loss_cls, 1)
    pc.add(rpn_loss_bbox, 1)
    pc.add(loss_cls, 1)
    pc.add(loss_bbox, 1)
    pc
  }
}
