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

package com.intel.analytics.bigdl.ppml.fgboost.common

import org.apache.logging.log4j.LogManager

import scala.collection.immutable.HashSet

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


  override def toString = s"TreeNode($nodeID, $isLeaf, score $similarScore, depth $depth, size ${recordSet.size})"
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
}


