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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.feature.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.FLContext
import com.intel.analytics.bigdl.ppml.base.Estimator
import com.intel.analytics.bigdl.ppml.fgboost.common.TreeUtils
import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto._
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto._
import com.intel.analytics.bigdl.ppml.utils.DataSetUtils
import com.intel.analytics.bigdl.ppml.utils.ProtoUtils._
import org.apache.logging.log4j.LogManager

import scala.util.control.Breaks.{break, breakable}
import scala.collection.mutable.ArrayBuffer
import collection.JavaConverters._
import collection.JavaConversions._
import scala.collection.mutable


class VflGBoostEstimator(continuous: Boolean,
                         nLabel: Int = 1,
                         learningRate: Float = 0.005f,
                         maxDepth: Int = 6,
                         minChildSize: Int = 1) extends Estimator {
  val logger = LogManager.getLogger(getClass)
  val flClient = FLContext.getClient()
  override protected val evaluateResults: mutable.Map[String, ArrayBuffer[Float]] = null
  val trees = new mutable.Queue[RegressionTree]()
  override def train(endEpoch: Int,
                     trainDataSet: LocalDataSet[MiniBatch[Float]],
                     valDataSet: LocalDataSet[MiniBatch[Float]]): Any = {
    // transform the LocalDataSet to Array type, the input type of RegressionTree
    val (feature, label) = DataSetUtils.localDataSetToArray(trainDataSet)
    val sortedIndexByFeature = TreeUtils.sortByFeature(feature)
    // TODO Load model from file
    // Sync VFL Worker/Client
    initFGBoost(label)

    if (continuous) {
      trainRegressionTree(feature, sortedIndexByFeature, endEpoch)
    } else {
      trainClassificationTree(feature, sortedIndexByFeature, endEpoch)
    }
  }

  override def evaluate(dataSet: LocalDataSet[MiniBatch[Float]]): Unit = {


  }

  override def predict(dataSet: LocalDataSet[MiniBatch[Float]]): Array[Activity] = {

    throw new NotImplementedError()
  }

  /**
   * Single round of tree boosting
   * @param roundId ID to upload to FLServer
   * @param tree the tree to boosting
   * @return true if the tree could continue boosting, false otherwise (if no leaves to split)
   */
  def boostRound(roundId: Int,
                 tree: RegressionTree): Boolean = {
    val i = roundId
    // Init an empty tree
    var st = System.currentTimeMillis()
    val currTree = tree
    logger.info("Tree Boost " + i.toString)
    buildTree(currTree, continuous = continuous)
    st = System.currentTimeMillis()
    logger.debug(s"Build tree cost ${(System.currentTimeMillis() - st) / 1000f} s")
    currTree.cleanup()
    // Add this tree into tree list
    logger.info(s"Built Tree_${i}" + currTree.toString)
    if (currTree.leaves.isEmpty) {
      logger.info("Early Stop boosting!")
      // Did not find leave dequeue usage, so this should never be called
      return false
    }
    // upload local tree
    val treeLeaves = currTree.leaves.toArray
    val treeIndexes = treeLeaves.map(_.nodeID.toInt).map(int2Integer).toList.asJava
    val treeOutput = treeLeaves.map(_.similarScore).map(float2Float).toList.asJava
    flClient.fgbostStub.uploadTreeLeaves(i.toString, treeIndexes, treeOutput)
    logger.debug(s"Update tree leaves ${(System.currentTimeMillis() - st) / 1000f} s")
    st = System.currentTimeMillis()
    trees.enqueue(currTree)
    // Evaluate tree and update residual and grads (g and h)
    validateLast(tree.dataset)
    logger.debug(s"Validate last ${(System.currentTimeMillis() - st) / 1000f} s")
    true
  }

  def trainRegressionTree(dataSet: Array[Tensor[Float]], indices: Array[Array[Int]], totalRound: Int): Unit = {
    for (i <- 0 until totalRound) {
      val grads = downloadGrad(i)
      val currTree = RegressionTree(dataSet, indices, grads, i.toString)
      val continueBoosting = boostRound(i, currTree)
      if (!continueBoosting) return
    }
  }
  def trainClassificationTree(dataSet: Array[Tensor[Float]], indices: Array[Array[Int]], totalRound: Int) = {
    val labelEarlyStop = new Array[Boolean](nLabel)
    for (i <- 0 until totalRound) {
      val grads = downloadGrad(i)
      val nGrads = TreeUtils.expandGrads(grads, dataSet.length, nLabel)
      for (gID <- 0 until nLabel) {
        if (!labelEarlyStop(gID)) {
          val currTree = RegressionTree(dataSet, indices, nGrads(gID), i.toString)
          val continueBoosting = boostRound(i, currTree)
          if (!continueBoosting) labelEarlyStop(gID) = true
        }
      }
    }
  }
  def buildTree(tree: RegressionTree, continuous: Boolean) = {
    while (tree.canGrow && tree.depth < maxDepth) {
      // Find best split in curr tree
      var st = System.currentTimeMillis()
      val bestLocalSplit = tree.findBestSplit()
      logger.debug(s"Find best split cost ${(System.currentTimeMillis() - st) / 1000f} s")
      st = System.currentTimeMillis()
      val bestSplit = getBestSplitFromServer(bestLocalSplit)
      logger.debug(s"Sync split cost ${(System.currentTimeMillis() - st) / 1000f} s")
      st = System.currentTimeMillis()
      val isLocalSplit = bestLocalSplit.getClientID == bestSplit.getClientID
      // If this split is in local dataset
      val updateCondition = if (continuous) {
        bestSplit.featureID == -1 || bestSplit.gain < 1e-6f
      } else  bestSplit.featureID == -1
      if (updateCondition) {
        logger.warn("Fail to split on current node")
        logger.info(s"Set Leaf gain = ${bestSplit.gain}")
        // Add current node to leaf
        tree.setLeaf(tree.nodes(bestSplit.nodeID))
        logger.debug(s"Set leaf ${(System.currentTimeMillis() - st) / 1000f} s")
        st = System.currentTimeMillis()
      } else {
        // Then, apply split & upload to VFL Server
        tree.updateTree(bestSplit, isLocalSplit)
        // End of building tree
        logger.debug(s"Update tree ${(System.currentTimeMillis() - st) / 1000f} s")
        st = System.currentTimeMillis()
      }
    }
  }

  /**
   * The initialization before boosting. Upload data label to FLServer
   * @param label
   */
  def initFGBoost(label: Array[Float]): Unit = {
    logger.info("Initializing VFL Boost...")
    // Init predict, grad & hess
    // only call in party with y
    val metadata = TableMetaData.newBuilder
      .setName(s"xgboost_grad").setVersion(0).build

    val gradData = Table.newBuilder.setMetaData(metadata)

    if (label != null && label.nonEmpty) {
      // party with label
      gradData.putTable("label", toFloatTensor(label))
    }
    // Upload
    flClient.fgbostStub.uploadTable(gradData.build)
  }

  /**
   * Download gradient from FLServer
   * @param treeID
   * @return
   */
  def downloadGrad(treeID: Int): Array[Array[Float]] = {
    // Note that g may be related to Y
    // H = 1 in regression
    val response = flClient.fgbostStub.downloadTable("xgboost_grad", treeID - 1)
    logger.info("Downloaded grads from FLServer")
    val gradTable = response.getData
    val grad = getTensor("grad", gradTable).toArray
    val hess = getTensor("hess", gradTable).toArray
    Array(grad, hess)
  }
  def getBestSplitFromServer(split: Split): Split = {
    split.setClientID(flClient.getClientUUID)
    val dataSplit = flClient.fgbostStub.split(split.toDataSplit()).getSplit

    Split(
      dataSplit.getTreeID,
      dataSplit.getNodeID,
      dataSplit.getFeatureID,
      dataSplit.getSplitValue,
      dataSplit.getGain,
      dataSplit.getItemSetList
    ).setClientID(dataSplit.getClientUid)
  }
  def validateLast(dataSet: Array[Tensor[Float]]): Unit = {
    // TODO: mini-batch support
    //    val batchSize = 100
    // predict with new tress
    logger.info("Eval new Tree")
    val localPredicts = dataSet.map { record =>
      Map(trees.last.treeID -> trees.last.predict(record))
    }
    val res = localPredicts.map{predict =>
      BoostEval.newBuilder()
        .addAllEvaluates(predict.toSeq.sortBy(_._1).map(p => {
          TreePredict.newBuilder().setTreeID(p._1)
            .addAllPredicts(p._2.map(boolean2Boolean).toList.asJava)
            .build()
        }).toList.asJava)
        .build()
    }.toList
    flClient.fgbostStub.evaluate(res.asJava)
  }
}
