package com.intel.analytics.bigdl.ppml.vfl.fgboost
import org.apache.logging.log4j.LogManager
import java.util

import com.intel.analytics.bigdl.dllib.tensor.Tensor

import scala.collection.JavaConversions.asScalaBuffer
import scala.collection.JavaConverters._
import scala.collection.mutable.HashSet
import scala.collection.mutable

class RegressionTree(
                      val dataset: Array[Tensor[Float]],
                      val sortedIndex: Array[Array[Int]],
                      val grads: Array[Array[Float]],
                      val treeID: String,
                      val numClasses: Int = 2,
                      var depth: Int = 0,
                      val minInstancesPerNode: Int = 8,
                      val numFeaturesPerNode: Int = 1,
                      val minInfoGain: Float = 0,
                      val lambda: Float = 1f
                    ) extends Serializable {


  val logger = LogManager.getLogger(this.getClass)

  var learningRate: Float = 0.3f

  // Store <nodeId, TreeNode>
  val nodes = new mutable.HashMap[String, TreeNode]
  // Store local split nodes
  val localNodes = new mutable.HashMap[String, Split]
  val leaves = new mutable.Queue[TreeNode]
  var groupID = 0
  protected var minChildSize = 1

  // Store splittable node queue
  val expandQueue = new mutable.Queue[TreeNode]

  init()

  def init(): Unit = {
    val rootNode = TreeNode("0",
      TreeUtils.computeScore(dataset.indices.toArray, grads, lambda),
      dataset.indices.toArray.toSet)
    expandQueue.enqueue(rootNode)
    nodes.put("0", rootNode)
  }

  def setLearningRate(lr: Float): this.type = {
    require(lr > 0, s"learning rate should greater than 0, but got ${lr}")
    learningRate = lr
    this
  }

  def setMinChildSize(size: Int): this.type = {
    require(size > 0, s"min child size should greater than 0, but got ${size}")
    minChildSize = size
    this
  }

  def findBestSplit(): Split = {
    logger.debug("Try to find best local split")
    val firstNode = expandQueue.dequeue()
    val bestLocalSplit = findBestSplitValue(firstNode)
    bestLocalSplit
  }

  def findBestSplitValue(treeNode: TreeNode): Split  = {
    // TODO: make minChildSize a parameter
    // For each feature
    val bestGainByFeature = sortedIndex.indices.par.map{fIndex =>
      val sortedFeatureIndex = sortedIndex(fIndex).filter(treeNode.recordSet.contains)
      // use Double to save gradSum, this will loss fewer precision.
      var leftGradSum = 0.0
      var leftHessSum  = 0.0
      var rightGradSum = sortedFeatureIndex.map(grads(0)(_)).sum.toDouble
      var rightHessSum = sortedFeatureIndex.map(grads(1)(_)).sum.toDouble
      var rStartIndex = 0
      var rEndIndex = minChildSize
      var fBestGain = minInfoGain
      var rBestIndex = -1
      while (rEndIndex < sortedFeatureIndex.length - minChildSize) {
        while (rEndIndex < sortedFeatureIndex.length &&
          dataset(sortedFeatureIndex(rEndIndex - 1)).valueAt(fIndex + 1) ==
            dataset(sortedFeatureIndex(rEndIndex)).valueAt(fIndex + 1)) {
          rEndIndex += 1
        }
        if (rEndIndex < sortedFeatureIndex.length - minChildSize) {
          val currGrad = sortedFeatureIndex.slice(rStartIndex, rEndIndex).map(grads(0)(_)).sum
          val currHess = sortedFeatureIndex.slice(rStartIndex, rEndIndex).map(grads(1)(_)).sum
          leftGradSum += currGrad
          leftHessSum += currHess
          val leftGain = TreeUtils.computeScoreWithSum(leftGradSum, leftHessSum, lambda)
          rightGradSum -= currGrad
          rightHessSum -= currHess
          val rightGain = TreeUtils.computeScoreWithSum(rightGradSum, rightHessSum, lambda)
          val currGain = leftGain + rightGain - treeNode.similarScore
          if (currGain > fBestGain) {
            fBestGain = currGain
            rBestIndex = rEndIndex
          }
          rStartIndex = rEndIndex
          rEndIndex += 1
        }
      }
      (fBestGain, fIndex, rBestIndex, sortedFeatureIndex)
    }
    val (bestGain, fIndex, rIndex, sortedFeatureIndex) = bestGainByFeature.maxBy(_._1)
    if (bestGain > minInfoGain) {
      require(rIndex > 0, s"best rIndex should greater than 0, but got ${rIndex}.")
      val leftSet = sortedFeatureIndex.slice(0, rIndex)
      // (leftMargin + rightMargin) / 2
      val splitValue = (dataset(sortedFeatureIndex(rIndex)).valueAt(fIndex + 1) +
        dataset(sortedFeatureIndex(rIndex - 1)).valueAt(fIndex + 1)) / 2
      val bestS = Split(
        treeID,
        treeNode.nodeID,
        fIndex,
        splitValue,
        bestGain,
        leftSet.map(int2Integer).toList.asJava
      )
      logger.info("Best local split on node " + treeNode.nodeID + " is " + bestS.toString)
      bestS
    } else {
      Split.leaf(treeID, treeNode.nodeID)
    }
  }


  def canGrow(): Boolean = {
    expandQueue.nonEmpty
  }


  def predict(record: Tensor[Float]): Array[Boolean] = {
    // Only predict with local nodes
    val res = Array.fill[Boolean](math.pow(2, depth + 1).toInt)(true)
    localNodes.values.foreach { split =>
      val currValue = record.valueAt(split.featureID + 1)
      res(split.nodeID.toInt) = currValue < split.splitValue
    }
    res
  }

  def updateTree(split: Split, isLocalSplit: Boolean): Unit = {
    if (!nodes.contains(split.nodeID)) {
      logger.info("Error%s" + nodes)
      logger.info(split)
    }
    val parentNode = nodes(split.nodeID)
    val newNodes = splitToNodes(split, parentNode)
    parentNode.leftChild = newNodes._1
    parentNode.rightChild = newNodes._2
    if (parentNode.rightChild.depth > depth) {
      depth = parentNode.rightChild.depth
    }
    if (parentNode.leftChild.recordSet.size <= numFeaturesPerNode) {
      setLeaf(parentNode.leftChild)
    } else {
      expandQueue.enqueue(parentNode.leftChild)
    }
    if (parentNode.rightChild.recordSet.size <= numFeaturesPerNode) {
      setLeaf(parentNode.rightChild)
    } else {
      expandQueue.enqueue(parentNode.rightChild)
    }
    nodes.put(parentNode.leftChild.nodeID, parentNode.leftChild)
    nodes.put(parentNode.rightChild.nodeID, parentNode.rightChild)
    if (isLocalSplit && split.featureID >= 0) {
      localNodes.put(parentNode.nodeID, split)
    }
  }

  def setLeaf(treeNode: TreeNode): Unit = {
    treeNode.isLeaf = true
    leaves.enqueue(treeNode)
    // Change to output
    treeNode.similarScore = learningRate *
      TreeUtils.computeOutput(treeNode.recordSet.toArray,
        grads, lambda)
  }

  def setGroup(gID: Int): Unit = {
    groupID = gID
  }

  def cleanup(): Unit = {
    // Set all nodes in queue as leaves
    while (expandQueue.nonEmpty) {
      setLeaf(expandQueue.dequeue)
    }
  }

  def splitToNodes(split: Split, treeNode: TreeNode): (TreeNode, TreeNode) = {
    val leftSet = split.itemSet.map(Integer2int).toSet
    val rightSet = treeNode.recordSet.diff(leftSet)
    // Left nodeID = parentNodeID * 2 + 1
    // Right nodeID = parentNodeID * 2 + 2
    // similarScore == output in regression
    val leftScore = TreeUtils.computeScore(leftSet.toArray, grads, lambda)
    val rightScore = TreeUtils.computeScore(rightSet.toArray, grads, lambda)
    val leftNode = TreeNode((treeNode.nodeID.toInt * 2 + 1).toString, leftScore, leftSet)
    val rightNode = TreeNode((treeNode.nodeID.toInt * 2 + 2).toString, rightScore, rightSet)
    leftNode.depth = treeNode.depth + 1
    rightNode.depth = treeNode.depth + 1
    (leftNode, rightNode)
  }

  def getDepth(): Int = {
    depth
  }

  override def toString = s"RegressionTree($treeID, depth $depth, local node $localNodes, leaves $leaves)"
}

object RegressionTree {

  def apply(treeID: String): RegressionTree = {
    new RegressionTree(Array(Tensor[Float](1, 2)),
      Array(Array(1, 2)), Array(Array(1, 2)), treeID)
  }

  def apply(dataset: Array[Tensor[Float]],
            sortedIndex: Array[Array[Int]],
            grads: Array[Array[Float]], treeID: String): RegressionTree = {
    new RegressionTree(dataset,
      sortedIndex, grads, treeID)
  }

}