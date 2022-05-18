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

import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer
import com.intel.analytics.bigdl.dllib.utils.Log4Error

case class XGBoostFormatNode(nodeid: Int,
                             depth: Int,
                             split: Int,
                             split_condition: Float,
                             leaf: Float,
                             yes: Int,
                             no: Int,
                             missing: Int,
                             children: List[XGBoostFormatNode])

class XGBoostSerializer {
  val jacksonJsonSerializer = new JacksonJsonSerializer()
  def deserialize(jsonStr: String): XGBoostFormatNode = {
    jacksonJsonSerializer.deSerialize(classOf[XGBoostFormatNode], jsonStr)
  }

}
object XGBoostFormatSerializer {
  val serializer = new XGBoostSerializer()
  def apply(jsonStr: String): XGBoostFormatNode = {
    serializer.deserialize(jsonStr)
  }
  def apply(regressionTree: RegressionTree): XGBoostFormatNode = {
    def buildXGBoostFormatTree(treeNode: TreeNode): XGBoostFormatNode = {
      if (treeNode.leftChild != null) {
        Log4Error.unKnowExceptionError(treeNode.rightChild != null,
          "???")
        XGBoostFormatNode(treeNode.nodeID.toInt,
          treeNode.depth,
          treeNode.splitInfo.featureID,
          treeNode.splitInfo.splitValue,
          0,
          treeNode.leftChild.nodeID.toInt,
          treeNode.rightChild.nodeID.toInt,
          missing = 0,
          // we do not need missing at this point because we will fiiNA thus no missing value
          List(buildXGBoostFormatTree(treeNode.leftChild),
            buildXGBoostFormatTree(treeNode.rightChild)))
      }
      else {
        // leaf node
        XGBoostFormatNode(treeNode.nodeID.toInt, 0, -1, 0, treeNode.similarScore,
          0, 0, 0, null)
      }
    }
    val rootNode = regressionTree.nodes.get("0").get
    buildXGBoostFormatTree(rootNode)
  }
}
