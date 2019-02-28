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

import scala.reflect.ClassTag

private[mkldnn] class Fusion {

  private val fuse = System.getProperty("bigdl.mkldnn.fusion", "false").toBoolean

  def fuseModule(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    if (!fuse) return;
    node.element match {
      case relu: ReLU =>
        fusionRelu(node)
      case bn: SpatialBatchNormalization =>
        fusionBn(node)
      case cadd: CAddTable =>
        fusionCAddTable(node)
      case _ =>
    }
  }

  /**
   * conv (relu false bn false) + bn
   * @param node
   */
  private def fusionBn(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    val bn = node.element.asInstanceOf[SpatialBatchNormalization]
    node.prevNodes.foreach(n => {
      n.element match {
        case conv : SpatialConvolution =>
          if (!conv.relu && !conv.batchNorm) {
            fusionConvBn(conv, bn)
            node.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]
          }
        case _ => null
      }})
  }

  /**
   *
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

  /**
   *
   * @param node
   */
  private def fusionCAddTable(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    if (node.element.isInstanceOf[CAddTable] && node.prevNodes.length == 2) {
      val previousNodes = node.prevNodes.toArray
      val layers = previousNodes.map(_.element)
      var conv : SpatialConvolution = null
      var other : Module[Float] = null
      // TODO: may not meet
      if (layers(0).isInstanceOf[SpatialConvolution]) {
        conv = layers(0).asInstanceOf[SpatialConvolution]
        other = layers(1)
      } else if (layers(1).isInstanceOf[SpatialConvolution]) {
        conv = layers(1).asInstanceOf[SpatialConvolution]
        other = layers(0)
      }
      // change the compution path
      if (other.isInstanceOf[ReLU] || other.isInstanceOf[SpatialBatchNormalization]) {
        if (other.isInstanceOf[ReLU]) {
          conv.setReLU()
        } else if (other.isInstanceOf[SpatialBatchNormalization]) {
          fusionConvBn(conv, other.asInstanceOf[SpatialBatchNormalization])
        }
        node.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]
      }
    }
  }

  private def fusionConvBn(conv: SpatialConvolution,
                           bn: SpatialBatchNormalization): Unit = {

    val originVar = Tensor[Float].resize(bn.runningVariance.size()).copy(bn.runningVariance.dense)
    val originMean = Tensor[Float].resize(bn.runningMean.size()).copy(bn.runningMean.dense)

    val convWeight = Tensor[Float].resize(conv.weight.size()).copy(conv.weight.dense)
    val convBias = Tensor[Float].resize(conv.bias.size()).copy(conv.bias.dense)

    (0 until bn.nOutput).foreach { j =>
      val variance = originVar.storage().array()(j + originVar.storageOffset() - 1)
      val base = Math.sqrt(variance.asInstanceOf[Float] + bn.eps).toFloat
      require(base != 0.0, s"the eps of ${bn.getName()} should be more than 0")

      val weight = if (conv.nGroup == 1) {
        convWeight.select(1, j + 1)
      } else {
        convWeight.select(2, j + 1)
      }
      weight.div(base)

      val bias = convBias.storage().array()(j)
      val mean = originMean.storage().array()(j)
      convBias.storage().array()(j) = (bias - mean) / base
    }

    conv.weight.copy(convWeight)
    conv.bias.copy(convBias)
  }
}
