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

import org.apache.logging.log4j.LogManager

import scala.collection.immutable.HashSet
import scala.util.parsing.json.{JSONArray, JSONObject}

/**
 *
 * @param nodeID the id of this node in tree
 * @param leftChild left child node TreeNode
 * @param rightChild right child node TreeNode
 * @param similarScore
 * @param recordSet the dataset this node could use to gain
 * @param depth
 */
class TreeNode (
                 val nodeID: String,
                 var leftChild: TreeNode,
                 var rightChild: TreeNode,
                 var similarScore: Float,
                 val recordSet: Set[Int],
                 var depth: Int = 0
               ) extends Serializable {

  val logger = LogManager.getLogger(this.getClass)
  var isLeaf = false
  var splitInfo: Split = null

  def canSplit(): Boolean = {
    ! isLeaf
  }

  def setLeaf(): Unit = {
    isLeaf = true
  }


  override def toString: String = s"TreeNode($nodeID," +
    s" $isLeaf, score $similarScore, depth $depth, size ${recordSet.size})"

  def toJSON(): JSONObject = {
    val map = scala.collection.mutable.Map[String, Any]()
    if (leftChild != null) {
      map("leftChild") = leftChild.nodeID
    }
    if (rightChild != null) {
      map("rightChild") = rightChild.nodeID
    }
    JSONObject(Map(
      "nodeID" -> nodeID,
      "similarScore" -> similarScore,
      "recordSet" -> JSONArray(recordSet.toList),
      "depth" -> depth,
      "isLeaf" -> isLeaf
    ) ++ map)
  }
}


object TreeNode {
  def apply(): TreeNode = {
    new TreeNode("1", null, null,
      0, HashSet[Int]())
  }

  def apply(nodeID: String): TreeNode = {
    new TreeNode(nodeID, null, null,
      0, HashSet[Int]())
  }

  def apply(nodeID: String,
            similarScore: Float,
            recordSet: Set[Int]
           ): TreeNode = {
    new TreeNode(nodeID, null, null, similarScore, recordSet)
  }
  def fromJson(json: JSONObject): TreeNode = {
    val nodeID = json.obj.get("nodeID").get.asInstanceOf[String]
    val similarScore = json.obj.get("similarScore").get.asInstanceOf[Double].toFloat
    val isLeaf = json.obj.get("isLeaf").get.asInstanceOf[Boolean]
    val recordSet = json.obj.get("recordSet").get.asInstanceOf[JSONArray].list
      .map(_.asInstanceOf[Double].toInt).toSet
    val treeNode = apply(nodeID, similarScore, recordSet)
    treeNode.depth = json.obj.get("depth").get.asInstanceOf[Double].toInt
    if (isLeaf) treeNode.setLeaf()
    treeNode
  }


}


