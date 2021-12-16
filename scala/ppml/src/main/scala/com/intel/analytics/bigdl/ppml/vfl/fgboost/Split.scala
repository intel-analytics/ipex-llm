/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.vfl.fgboost

import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DataSplit
import org.apache.log4j.Logger
import java.util


class Split (
              val treeID: String,
              val nodeID: String,
              val featureID: Int,
              val splitValue: Float,
              val gain: Float,
              val itemSet: util.List[Integer]
            ) extends Serializable {
  val logger = Logger.getLogger(this.getClass)

  protected var clientID = "";

  def getClientID: String= clientID

  def setClientID(clientID: String): this.type = {
    this.clientID = clientID
    this
  }

  override def toString = s"Split($clientID, $treeID, $nodeID, $featureID, $splitValue, gain $gain, left size ${itemSet.size})"


  def canEqual(other: Any): Boolean = other.isInstanceOf[Split]

  override def equals(other: Any): Boolean = other match {
    case that: Split =>
      (that canEqual this) &&
        treeID == that.treeID &&
        nodeID == that.nodeID &&
        featureID == that.featureID &&
        splitValue == that.splitValue &&
        gain == that.gain &&
        itemSet == that.itemSet
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(treeID, nodeID, featureID, splitValue, gain, itemSet)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  def toDataSplit(): DataSplit = {
    DataSplit.newBuilder
      .setTreeID(treeID)
      .setNodeID(nodeID)
      .setFeatureID(featureID)
      .setSplitValue(splitValue)
      .setGain(gain)
      .setSetLength(itemSet.size())
      .setClientUid(clientID)
      .addAllItemSet(itemSet).build
  }
}

object Split {
  def fromDataSplit(dataSplit: DataSplit): Split = {
    apply(dataSplit.getTreeID,
      dataSplit.getNodeID,
      dataSplit.getFeatureID,
      dataSplit.getSplitValue,
      dataSplit.getGain,
      dataSplit.getItemSetList).setClientID(dataSplit.getClientUid)
  }

  def leaf(treeID: String,
           nodeID: String): Split = {
    apply(treeID, nodeID, -1, -1, 0, new util.ArrayList[Integer]())
  }

  def apply(treeID: String,
            nodeID: String,
            featureID: Int,
            splitValue: Float,
            gain: Float, bitSet: util.List[Integer]): Split = {
    new Split(treeID, nodeID, featureID, splitValue, gain, bitSet)
  }
}


