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

package com.intel.analytics.bigdl.ppml.fl.fgboost.common

import com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DataSplit
import org.apache.logging.log4j.LogManager

import java.util
import scala.util.parsing.json.{JSONArray, JSONObject}
import scala.collection.JavaConverters._

class Split (
              val treeID: String,
              val nodeID: String,
              val featureID: Int,
              val splitValue: Float,
              val gain: Float,
              val itemSet: util.List[Integer]
            ) extends Serializable {
  val logger = LogManager.getLogger(this.getClass)
  protected var version = -1
  protected var clientID = "";
  protected var featureName = "";
  def getClientID: String = clientID

  def setVersion(version: Int): Split = {
    this.version = version
    this
  }
  def getFeatureName(): String = featureName
  def setFeatureName(featureName: String): Split = {
    this.featureName = featureName
    this
  }
  def setClientID(clientID: String): this.type = {
    this.clientID = clientID
    this
  }

  override def toString: String = s"Split($clientID," +
    s" $treeID, $nodeID, $featureID, $splitValue, gain $gain, left size ${itemSet.size})"


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
      .setVersion(version)
      .addAllItemSet(itemSet).build
  }

  def toJSON(): JSONObject = {
    JSONObject(Map("treeID" -> treeID,
      "nodeID" -> nodeID,
      "featureID" -> featureID,
      "splitValue" -> splitValue,
      "gain" -> gain,
      "itemSet" -> JSONArray(itemSet.asScala.toList)
    ))
  }
}

object Split {
  def fromDataSplit(dataSplit: DataSplit): Split = {
    apply(dataSplit.getTreeID,
      dataSplit.getNodeID,
      dataSplit.getFeatureID,
      dataSplit.getSplitValue,
      dataSplit.getGain,
      dataSplit.getItemSetList,
      dataSplit.getVersion).setClientID(dataSplit.getClientUid)
  }

  def leaf(treeID: String,
           nodeID: String): Split = {
    apply(treeID, nodeID, -1, -1, 0, new util.ArrayList[Integer](), 0)
  }

  def apply(treeID: String,
            nodeID: String,
            featureID: Int,
            splitValue: Float,
            gain: Float,
            bitSet: util.List[Integer],
            version: Int = -1): Split = {
    new Split(treeID, nodeID, featureID, splitValue, gain, bitSet).setVersion(version)

  }
  def fromJson(json: JSONObject): Split = {
    val treeID = json.obj.get("treeID").get.asInstanceOf[String]
    val nodeID = json.obj.get("nodeID").get.asInstanceOf[String]
    val featureID = json.obj.get("featureID").get.asInstanceOf[Double].toInt
    val splitValue = json.obj.get("splitValue").get.asInstanceOf[Double].toFloat
    val gain = json.obj.get("gain").get.asInstanceOf[Double].toFloat
    val itemSet = json.obj.get("itemSet").get.asInstanceOf[JSONArray].list
      .map(_.asInstanceOf[Double].toInt).map(int2Integer)
    apply(treeID, nodeID, featureID, splitValue, gain, itemSet.asJava)
  }
}


