package com.intel.analytics.bigdl.ppml.fgboost.common

import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer


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
  def deserialize(jsonStr: String) = {
    jacksonJsonSerializer.deSerialize(classOf[XGBoostFormatNode], jsonStr)
  }

}
object XGBoostFormatSerializer {
  val serializer = new XGBoostSerializer()
  def apply(jsonStr: String) = {
    serializer.deserialize(jsonStr)
  }
  def apply(regressionTree: RegressionTree) = {
    def buildXGBoostFormatTree(treeNode: TreeNode): XGBoostFormatNode = {
      if (treeNode.leftChild != null) {
        require(treeNode.rightChild != null, "???")
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
        XGBoostFormatNode(treeNode.nodeID.toInt, 0, 0, 0, treeNode.similarScore,
          0, 0, 0, null)
      }
    }
    val rootNode = regressionTree.nodes.get("0").get
    buildXGBoostFormatTree(rootNode)
  }
}