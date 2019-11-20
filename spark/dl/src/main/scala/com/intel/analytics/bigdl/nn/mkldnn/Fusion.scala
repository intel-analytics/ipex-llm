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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.{MklInt8Convertible, Scale => ScaleLayer}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Node

/**
 * Add fusion operation for dnn graph node, there are three cases about fusion:
 * case 1: fuse relu with conv(SpatialConvolution) or bn(SpatialBatchNormalization)
 * case 2: fuse conv with bn
 * case 3: sum conv output with another layer output
 * If you want to use fusion for inference, please set property "bigdl.mkldnn.fusion" to true
 */
private[mkldnn] object Fusion {

  private def fuse = System.getProperty("bigdl.mkldnn.fusion", "true").toBoolean

  def fuseModule(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    if (!fuse) return;
    node.element match {
      case relu: ReLU => fusionRelu(node)
      case bn: SpatialBatchNormalization => fusionBN(node)
      case _ =>
    }
  }

  def fuseCAdd(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    if (!fuse) return;
    node.element match {
      case cadd: CAddTable => fusionCAddTable(node)
      case _ =>
    }
  }

  /**
   * fuse conv(without relu or bn fusion) with bn
   * if bn has fused with relu, then fuse relu and bn with conv
   * @param node
   */
  private def fusionBN(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    val bn = node.element.asInstanceOf[SpatialBatchNormalization]
    node.prevNodes.foreach(n => {
      n.element match {
        case conv : SpatialConvolution =>
          // reminder: may be conv can fuse with two bn
          if (!conv.relu && !conv.batchNorm) {
            if (bn.relu) conv.setReLU(true)
            fusionConvBn(conv, bn)
            node.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]
          }
        case _ => null
      }})
  }

  /**
   * fuse relu with conv or bn
   * @param node
   */
  private def fusionRelu(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    node.prevNodes.foreach(n => {
      val notIdentity = findPrevious(n)
      notIdentity.element match {
        case conv: SpatialConvolution =>
          if (!conv.relu) {
            conv.setReLU(true)
            conv.setOutputScales(node.element.asInstanceOf[ReLU].getOutputScales())
            node.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]
          }
        case bn: SpatialBatchNormalization =>
          if (!bn.relu) {
            bn.setReLU(true)
            bn.setOutputScales(node.element.asInstanceOf[ReLU].getOutputScales())
            node.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]
          }
        case _ => null
      }})
  }

  private def findPrevious(node: Node[AbstractModule[Activity, Activity, Float]])
  : Node[AbstractModule[Activity, Activity, Float]] = {
    if (node.element.isInstanceOf[Identity] && node.prevNodes.length == 1) {
      findPrevious(node.prevNodes(0))
    } else node
  }

  private def findNext(node: Node[AbstractModule[Activity, Activity, Float]])
  : Seq[Node[AbstractModule[Activity, Activity, Float]]] = {
    if (node.element.isInstanceOf[Identity]) {
      node.nextNodes.flatMap(n => findNext(n))
    } else {
      Seq(node)
    }
  }

  /**
   * If previous layers number of CAddTable is two, and one of it is conv layer.
   * then fuse output of the other layer in conv layer.
   * @param node
   */
  private def fusionCAddTable(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    if (node.element.isInstanceOf[CAddTable] && node.prevNodes.length == 2) {
      val previousNodes = node.prevNodes.toArray
      val node1 = findPrevious(previousNodes(0))
      val node2 = findPrevious(previousNodes(1))

      var conv : Node[Module[Float]] = null
      var otherNumber: Int = 0

      if (node1.element.isInstanceOf[SpatialConvolution]) {
        if (requirements(node1)) conv = node1
        otherNumber = 1
      } else if (node2.element.isInstanceOf[SpatialConvolution]) {
        if (requirements(node2)) conv = node2
        otherNumber = 0
      }
      // meet fuse requirements
      if (conv != null) {
        node.element = conv.element
        val element = node.element.asInstanceOf[SpatialConvolution]
        element.setSumOp(previousNodes(otherNumber).element, otherNumber + 1)
        conv.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]

        val nexts = node.nextNodes(0)
        if (nexts.element.isInstanceOf[ReLU] && !element.relu) {
          node.element.asInstanceOf[SpatialConvolution].setReLU(true)
          node.element.asInstanceOf[SpatialConvolution].setOutputScales(
            nexts.element.asInstanceOf[ReLU].getOutputScales())
          nexts.element = new Identity()
        }

        val prevIsNotIdentity = findPrevious(previousNodes(otherNumber))

        prevIsNotIdentity.element match {
          case conv: SpatialConvolution =>
            conv.setOutputScales(node.element.asInstanceOf[SpatialConvolution].getOutputScales())
          case relu: ReLU =>
            relu.setOutputScales(node.element.asInstanceOf[SpatialConvolution].getOutputScales())
            prevIsNotIdentity.nextNodes.flatMap(x => findNext(x))
              .filter(x => x != node && x.element.isInstanceOf[MklInt8Convertible])
              .foreach(_.element.asInstanceOf[MklInt8Convertible].setInputScales(
                node.element.asInstanceOf[SpatialConvolution].getOutputScales()))
          case _ =>
        }
      }
    }
  }

  private def requirements(node: Node[AbstractModule[Activity, Activity, Float]]): Boolean = {
    val conv = node.element.asInstanceOf[SpatialConvolution]
    if (conv.sum) false else true
  }

  private def fusionConvBn(conv: SpatialConvolution,
                           bn: SpatialBatchNormalization): Unit = {
    conv.setBatchNorm(true)
    val originVar = Tensor[Float].resize(bn.runningVariance.size()).copy(bn.runningVariance.dense)
    val originMean = Tensor[Float].resize(bn.runningMean.size()).copy(bn.runningMean.dense)

    val convWeight = Tensor[Float].resize(conv.weight.size()).copy(conv.weight.dense)
    val convBias = Tensor[Float].resize(conv.bias.size()).copy(conv.bias.dense)

    val bnWeight = Tensor[Float].resizeAs(bn.weightAndBias.dense).copy(bn.weightAndBias.dense)

    (0 until bn.nOutput).foreach { j =>
      val variance = originVar.storage().array()(j + originVar.storageOffset() - 1)
      val base = Math.sqrt(variance.asInstanceOf[Float] + bn.eps).toFloat
      require(base != 0.0, s"the eps of ${bn.getName()} should be more than 0")

      val alpha = bnWeight.storage().array()(bnWeight.storageOffset() - 1 + j)
      val beta = bnWeight.storage().array()(bnWeight.storageOffset() - 1 + bn.nOutput + j)

      val weight = if (conv.nGroup == 1) {
        convWeight.select(1, j + 1)
      } else {
        val channelPerGroup = conv.nOutputPlane / conv.nGroup
        val group = j  / channelPerGroup + 1
        val channel = j % channelPerGroup + 1
        convWeight.select(1, group).select(2, channel)
      }
      weight.div(base)
      weight.mul(alpha)

      val bias = convBias.storage().array()(j)
      val mean = originMean.storage().array()(j)
      convBias.storage().array()(j) = alpha / base * bias + beta - (alpha * mean) / base
    }

    // We will change model structure and weights when doing conv and bn fusion
    // In order to not influence broadcast model and weights,
    // we set new storage to weight and bias.
    conv.weight.dense.set(convWeight)
    conv.bias.dense.set(convBias)

    // regenerate the weight scales and output scales
    conv.flushWeightScales(conv.weight.dense)
    conv.setOutputScales(bn.getOutputScales())
  }

  def setNegativeInputOfConv(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {

    if (!fuse || !node.element.isInstanceOf[SpatialConvolution]) return

    val successFromReLU = node.prevNodes.flatMap(x => findAllNonIdentityPrevs(x))
      .map { x =>
        x.element match {
          case _: SpatialConvolution =>
            x.element.asInstanceOf[SpatialConvolution].relu
          case _: ReLU =>
            true
          case _ =>
            false
        }
      }.forall(_ == true)


    if (successFromReLU) {
      node.element.asInstanceOf[SpatialConvolution].negativeInput = false
    }
  }

  /**
   * set the layers' scales which is previous nodes of JoinTable.
   *
   * For a graph structure like below,
   *
   * conv1 --+
   *         |--> JoinTable --> conv3
   * conv2 --+
   *
   * we should set the conv1's and conv2's output scales to the conv3's input scales.
   *
   * If the operation next JoinTable has no input scales like below. We should set
   * the scales to the max values of input scales of conv1 and conv2.
   *
   * conv1 --+
   *         |--> JoinTable --> [Layer/Op has no input scales]
   * conv2 --+
   *
   * @param node current node
   */
  def setScalesPrevousJoinTable(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    // case 1, need not do fusion
    if (!fuse || !node.element.isInstanceOf[JoinTable]) return

    val preConvs = node.prevNodes.flatMap(x => findAllNonIdentityPrevs(x))
      .filter(_.element.isInstanceOf[SpatialConvolution])
      .map(_.element.asInstanceOf[SpatialConvolution])

    // case 2, there's one node need not quantize
    if (!preConvs.exists(_.needQuantize)) return

    // case 3, the output dimension mask should be the same
    val masks = preConvs.map(_.getOutputDimMask()).toSet
    require(masks.size == 1, s"all preceding convolutions must have the same mask")

    val nextConvs = node.nextNodes.flatMap(findNext)
      .filter(_.element.isInstanceOf[SpatialConvolution])

    val scales = if (nextConvs.isEmpty) {
      Array(preConvs.map(_.getOutputScales().flatten).transpose.map(_.max).toArray)
    } else {
      nextConvs.map(_.element.asInstanceOf[SpatialConvolution]).head.getInputScales()
    }

    preConvs.foreach { conv =>
      conv.setOutputScales(scales)
    }
  }

  def fuseScale(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    node.element match {
      case wrapper: BlasWrapper if wrapper.module.isInstanceOf[ScaleLayer[Float]] =>
        // check all prevNodes are SpatialBatchNormalization
        val isValid = node.prevNodes.forall(_.element.isInstanceOf[SpatialBatchNormalization])
        if (!isValid) { return }

        node.prevNodes.foreach { prevNode =>
          val bn = prevNode.element.asInstanceOf[SpatialBatchNormalization]
          val weightAndBias = bn.weightAndBias.dense
          val weight = weightAndBias.narrow(1, 1, bn.nOutput)
          val bias = weightAndBias.narrow(1, bn.nOutput + 1, bn.nOutput)

          val scale = node.element.asInstanceOf[BlasWrapper].module.asInstanceOf[ScaleLayer[Float]]
          val scaleWeight = scale.parameters()._1(0)
          val scaleBias = scale.parameters()._1(1)

          weight.cmul(scaleWeight)
          bias.cmul(scaleWeight)
          bias.add(scaleBias)


          // set the weight and bias to new tensor, we do not modify the original model's tensor.
          // sometimes, the model need to be reused.
          bn.weightAndBias.dense.set(weightAndBias)
        }

        node.element = Identity[Float]() // set the BlasWrapper to Identity, we need no scale now
      case _ =>
    }
  }

  private def findAllNonIdentityPrevs(node: Node[AbstractModule[Activity, Activity, Float]])
  : Seq[Node[AbstractModule[Activity, Activity, Float]]] = {
    // TODO currently, it will only skip the Identity, MaxPooling, AvgPooling, JoinTable
    // becase if the output of layer/op previous of the four, they will output
    // nonnegative too. it's not an elegant impl.
    if (node.element.isInstanceOf[Identity] ||
      node.element.isInstanceOf[MaxPooling] ||
      node.element.isInstanceOf[AvgPooling] ||
      node.element.isInstanceOf[JoinTable]) {
      node.prevNodes.flatMap(findAllNonIdentityPrevs)
    } else {
      Seq(node)
    }
  }
}
