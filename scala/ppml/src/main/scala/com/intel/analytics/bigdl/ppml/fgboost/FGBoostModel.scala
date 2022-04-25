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

package com.intel.analytics.bigdl.ppml.fgboost

import com.intel.analytics.bigdl.dllib.optim.ValidationMethod
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.FLContext
import com.intel.analytics.bigdl.ppml.fgboost.common.{RegressionTree, Split, TreeUtils}
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.{MetaData, TensorMap}
import com.intel.analytics.bigdl.ppml.utils.DataFrameUtils
import com.intel.analytics.bigdl.ppml.utils.ProtoUtils.{getTensor, toArrayFloat, toBoostEvals, toFloatTensor}
import org.apache.logging.log4j.LogManager

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import collection.JavaConverters._

abstract class FGBoostModel(continuous: Boolean,
                            nLabel: Int = 1,
                            learningRate: Float = 0.005f,
                            maxDepth: Int = 6,
                            minChildSize: Int = 1,
                            validationMethods: Array[ValidationMethod[Float]] = null) {
  val logger = LogManager.getLogger(getClass)
  var flClient = FLContext.getClient()

  protected val evaluateResults: mutable.Map[String, ArrayBuffer[Float]] = null
  val trees = new mutable.Queue[RegressionTree]()
  def fit(feature: Array[Tensor[Float]],
          label: Array[Float],
          boostRound: Int) = {
    val sortedIndexByFeature = TreeUtils.sortByFeature(feature)
    // TODO Load model from file
    initFGBoost(label)

    if (continuous) {
      trainRegressionTree(feature, sortedIndexByFeature, boostRound)
    } else {
      trainClassificationTree(feature, sortedIndexByFeature, boostRound)
    }
  }
  def evaluate(feature: Array[Tensor[Float]],
               label: Array[Float]) = {
    val predictResult = predictTree(feature)
    val predictActivity = Tensor[Float](predictResult, Array(predictResult.length))
    val targetProto = flClient.fgbostStub.downloadLabel("label", 0).getData
    val targetActivity = getTensor("label", targetProto)
    validationMethods.map(vMethod => {
      vMethod.apply(predictActivity, targetActivity)
    })
  }
  def predict(feature: Array[Tensor[Float]]) = {
    val predictResult = predictTree(feature)
    predictResult.map{ value =>
      Tensor[Float](Array(value), Array(1))
    }
  }
  /**
   * Use server tree to predict input
   * @param inputs the input data
   * @return predict result
   */
  def predictTree(inputs: Array[Tensor[Float]]): Array[Float] = {
    val localPredicts = inputs.map { record =>
      trees.indices.map(i =>
        trees(i).treeID -> trees(i).predict(record)).toMap
    }
    val booleanOnePredict = localPredicts.head.values.map(_.length).sum
    // message may be too large, split by group to send to FLServer
    val messageSize = 2 * 1e6
    val groupedSize = Math.ceil(messageSize / booleanOnePredict).toInt
    val result = localPredicts.grouped(groupedSize).flatMap{ groupedPredicts =>
      val boostEvals = toBoostEvals(groupedPredicts)
      val response = flClient.fgbostStub.predict(boostEvals.asJava)
      toArrayFloat(response)
    }.toArray
    result
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
      logger.info("No leaves could be expanded, early Stop boosting.")
      return false
    }
    // upload local tree
    val treeLeaves = currTree.leaves.toArray
    val treeIndexes = treeLeaves.map(_.nodeID.toInt).map(int2Integer).toList.asJava
    val treeOutput = treeLeaves.map(_.similarScore).map(float2Float).toList.asJava
    flClient.fgbostStub.uploadTreeLeaf(i.toString, treeIndexes, treeOutput)
    logger.debug(s"Update tree leaves ${(System.currentTimeMillis() - st) / 1000f} s")
    st = System.currentTimeMillis()
    trees.enqueue(currTree)
    // Evaluate tree and update residual and grads (g and h)
    uploadResidual(tree.dataset)
    logger.debug(s"Upload residual of last tree ${(System.currentTimeMillis() - st) / 1000f} s")
    true
  }

  /**
   * Use local tree to predict, and upload residual to FLServer
   * @param data the input data to predict
   * @return
   */
  def uploadResidual(data: Array[Tensor[Float]]) = {
    val lastTreePredict = data.map { record =>
      Map(trees.last.treeID -> trees.last.predict(record))
    }
    val boostEvals = toBoostEvals(lastTreePredict)
    // TODO: add grouped sending message
    flClient.fgbostStub.evaluate(boostEvals.asJava)
  }


  def trainRegressionTree(dataSet: Array[Tensor[Float]], indices: Array[Array[Int]], totalRound: Int): Unit = {
    for (i <- 0 until totalRound) {
      logger.debug(s"Training regression tree boost round: $i")
      val grads = downloadGrad(i)
      val currTree = RegressionTree(dataSet, indices, grads, i.toString)
      currTree.setLearningRate(learningRate).setMinChildSize(minChildSize)
      val continueBoosting = boostRound(i, currTree)
      if (!continueBoosting) return
    }
  }
  def trainClassificationTree(dataSet: Array[Tensor[Float]], indices: Array[Array[Int]], totalRound: Int) = {
    val labelEarlyStop = new Array[Boolean](nLabel)
    for (i <- 0 until totalRound) {
      logger.debug(s"Training classification tree boost round: $i")
      val grads = downloadGrad(i)
      val nGrads = TreeUtils.expandGrads(grads, dataSet.length, nLabel)
      for (gID <- 0 until nLabel) {
        if (!labelEarlyStop(gID)) {
          val currTree = RegressionTree(dataSet, indices, nGrads(gID), i.toString)
          currTree.setLearningRate(learningRate).setMinChildSize(minChildSize)
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
        // update bestSplit from server to local tree
        tree.updateTree(bestSplit, isLocalSplit)
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
    val metadata = MetaData.newBuilder
      .setName(s"xgboost_grad").setVersion(0).build

    val gradData = TensorMap.newBuilder.setMetaData(metadata)

    if (label != null && label.nonEmpty) {
      // party with label
      gradData.putTensors("label", toFloatTensor(label))
    }
    // Upload
    flClient.fgbostStub.uploadLabel(gradData.build)
  }

  /**
   * Download gradient from FLServer
   * @param treeID
   * @return
   */
  def downloadGrad(treeID: Int): Array[Array[Float]] = {
    // Note that g may be related to Y
    // H = 1 in regression
    val response = flClient.fgbostStub.downloadLabel("xgboost_grad", treeID - 1)
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
}
