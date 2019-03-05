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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Node

private[mkldnn] object Fusion {

  private val fuse = System.getProperty("bigdl.mkldnn.fusion", "false").toBoolean

  def fuseModule(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    if (!fuse) return;
    node.element match {
      case relu: ReLU => fusionRelu(node)
      case bn: SpatialBatchNormalization => fusionBn(node)
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
  private def fusionBn(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    val bn = node.element.asInstanceOf[SpatialBatchNormalization]
    node.prevNodes.foreach(n => {
      n.element match {
        case conv : SpatialConvolution =>
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
      n.element match {
        case conv: SpatialConvolution =>
          if (!conv.relu) {
            conv.setReLU(true)
            node.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]
          }
        case bn: SpatialBatchNormalization =>
          if (!bn.relu) {
            bn.setReLU(true)
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
        element.setSumOp(previousNodes(otherNumber - 1).element, otherNumber)
        conv.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]

        val nexts = node.nextNodes(0)
        if (nexts.element.isInstanceOf[ReLU] && !element.relu) {
          node.element.asInstanceOf[SpatialConvolution].setReLU(true)
          nexts.element = new Identity()
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
        convWeight.select(2, j + 1)
      }
      weight.div(base)
      weight.mul(alpha)

      val bias = convBias.storage().array()(j)
      val mean = originMean.storage().array()(j)
      convBias.storage().array()(j) = alpha / base * bias + beta - (alpha * mean) / base
    }

    conv.weight.copy(convWeight)
    conv.bias.copy(convBias)
  }
}
