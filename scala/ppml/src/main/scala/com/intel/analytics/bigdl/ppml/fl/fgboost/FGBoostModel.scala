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

package com.intel.analytics.bigdl.ppml.fl.fgboost

import com.intel.analytics.bigdl.dllib.optim.{ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.fl.fgboost.common.{RegressionTree, Split, TreeUtils}
import com.intel.analytics.bigdl.ppml.fl.FLContext
import com.intel.analytics.bigdl.ppml.fl.generated.FlBaseProto.{MetaData, TensorMap}
import com.intel.analytics.bigdl.ppml.fl.utils.{DataFrameUtils, FLClientClosable}
import com.intel.analytics.bigdl.ppml.fl.utils.ProtoUtils.{getTensor, toArrayFloat, toBoostEvals, toFloatTensor}
import jdk.nashorn.internal.ir.debug.ObjectSizeCalculator
import com.intel.analytics.bigdl.ppml.fl.utils.Conventions._
import org.apache.logging.log4j.LogManager

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import collection.JavaConverters._
import scala.util.parsing.json.{JSON, JSONObject}

abstract class FGBoostModel(continuous: Boolean,
                            nLabel: Int = 1,
                            learningRate: Float = 0.005f,
                            maxDepth: Int = 6,
                            minChildSize: Int = 1,
                            validationMethods: Array[ValidationMethod[Float]] = null,
                            serverModelPath: String = null)
  extends FLClientClosable {
  val logger = LogManager.getLogger(getClass)
  var splitVersion = 0
  var treeLeafVersion = 0
  var evaluateVersion = 0
  var predictVersion = 0
  protected val evaluateResults: mutable.Map[String, ArrayBuffer[Float]] = null
  var xTrainBuffer: ArrayBuffer[Tensor[Float]] = new ArrayBuffer[Tensor[Float]]()
  val trees = new mutable.Queue[RegressionTree]()
  var curLoss: Float = Float.MaxValue

  def loadServerModel(modelPath: String): Unit = {
    flClient.fgbostStub.loadServerModel(modelPath)
  }
  def fit(feature: Array[Tensor[Float]],
          label: Array[Float],
          boostRound: Int): Unit = {
    val sortedIndexByFeature = TreeUtils.sortByFeature(feature)
    // TODO Load model from file
    initFGBoost(label)
    if (trees.nonEmpty) {
      logger.info(s"This is incremental training, evaluating using existed trees..")
      (0 until trees.size).foreach(i => uploadResidual(feature, i))

    }

    if (continuous) {
      trainRegressionTree(feature, sortedIndexByFeature, boostRound)
    } else {
      trainClassificationTree(feature, sortedIndexByFeature, boostRound)
    }
  }
  def fitAdd(xTrainBatch: Array[Tensor[Float]]): ArrayBuffer[Tensor[Float]] = {
    xTrainBuffer ++= xTrainBatch
  }
  def fitCall(yTrain: Array[Float], boostRound: Int): Unit = {
    val xTrain = xTrainBuffer.toArray
    logger.info(s"start to sort index")
    val sortedIndexByFeature = TreeUtils.sortByFeature(xTrain)
    logger.info(s"sort index end")
    // TODO Load model from file
    initFGBoost(yTrain)
    if (trees.nonEmpty) {
      logger.info(s"This is incremental training, evaluating using existed trees..")
      (0 until trees.size).foreach(i => uploadResidual(xTrain, i))
    }
    if (continuous) {
      trainRegressionTree(xTrain, sortedIndexByFeature, boostRound)
    } else {
      trainClassificationTree(xTrain, sortedIndexByFeature, boostRound)
    }
  }


  def evaluate(feature: Array[Tensor[Float]],
               label: Array[Float]): Array[ValidationResult] = {
    val predictResult = predictTree(feature)
    val predictActivity = Tensor[Float](predictResult, Array(predictResult.length))
    val targetProto = flClient.fgbostStub.downloadLabel("label", 0).getData
    val targetActivity = getTensor("label", targetProto)
    validationMethods.map(vMethod => {
      vMethod.apply(predictActivity, targetActivity)
    })
  }
  def predict(feature: Array[Tensor[Float]]): Array[Tensor[Float]] = {
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
    //    val messageSize = 2 * 1e6
    //    val groupedSize = Math.ceil(messageSize / booleanOnePredict).toInt
    //    val result = localPredicts.grouped(groupedSize).flatMap{ groupedPredicts =>
    //      val boostEvals = toBoostEvals(groupedPredicts)
    //      val response = flClient.fgbostStub.predict(boostEvals.asJava)
    //      toArrayFloat(response)
    //    }.toArray

    val boostEvals = toBoostEvals(localPredicts)
    val response = flClient.fgbostStub.predict(boostEvals.asJava, predictVersion)
    predictVersion += 1
    val result = toArrayFloat(response)
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
    val currTree = tree
    buildTree(currTree, continuous = continuous)
    currTree.cleanup()
    // Add this tree into tree list
    //    logger.info(s"Built Tree_${i}" + currTree.toString)
    if (currTree.leaves.isEmpty) {
      logger.info("No leaves could be expanded, early Stop boosting.")
      return false
    }
    // upload local tree
    val treeLeaves = currTree.leaves.toArray
    val treeIndexes = treeLeaves.map(_.nodeID.toInt).map(int2Integer).toList.asJava
    val treeOutput = treeLeaves.map(_.similarScore).map(float2Float).toList.asJava
    flClient.fgbostStub.uploadTreeLeaf(i.toString, treeIndexes, treeOutput, evaluateVersion)
    evaluateVersion += 1
    trees.enqueue(currTree)
    // Evaluate tree and update residual and grads (g and h)
    uploadResidual(tree.dataset)
    if (serverModelPath != null) {
      flClient.fgbostStub.saveServerModel(serverModelPath)
    }
    true
  }

  /**
   * Use local tree to predict, and upload residual to FLServer
   * @param data the input data to predict
   * @return
   */
  def uploadResidual(data: Array[Tensor[Float]], idx: Int = trees.size - 1): Unit = {
    logger.info(s"Uploading tree $idx ...")
    val predictToUpload = data.map { record =>
      val tree = trees.get(idx).get
      Map(tree.treeID -> tree.predict(record))
    }

    val boostEvals = toBoostEvals(predictToUpload)
    // TODO: add grouped sending message

    val perMsgSize = ObjectSizeCalculator.getObjectSize(boostEvals.head)
    val dataPerGroup = MAX_MSG_SIZE / perMsgSize
    logger.debug(s"data num: ${boostEvals.size}," +
      s" per msg size: $perMsgSize, data per group: $dataPerGroup")
    var sended = 0
    var lastBatch = false
    boostEvals.grouped(dataPerGroup.toInt).foreach(l => {
      if (sended + dataPerGroup.toInt >= boostEvals.size) lastBatch = true
      logger.info(s"evaluating in train step, version: $evaluateVersion")
      val response = flClient.fgbostStub.evaluate(l.asJava, evaluateVersion, lastBatch)
      logger.info(response.getResponse)
      sended += l.size
    })


    //    flClient.fgbostStub.evaluate(boostEvals.asJava, evaluateVersion)
  }


  def trainRegressionTree(dataSet: Array[Tensor[Float]],
                          indices: Array[Array[Int]], totalRound: Int): Unit = {
    for (i <- 0 until totalRound) {
      val grads = downloadGrad(i)
      logger.info(s"Round: $i/$totalRound, loss: $curLoss")
      val currTree = RegressionTree(dataSet, indices, grads, i.toString)
      currTree.init()
      currTree.setLearningRate(learningRate).setMinChildSize(minChildSize)
      val continueBoosting = boostRound(i, currTree)
      if (!continueBoosting) return
    }
  }
  def trainClassificationTree(dataSet: Array[Tensor[Float]],
                              indices: Array[Array[Int]], totalRound: Int): Unit = {
    val labelEarlyStop = new Array[Boolean](nLabel)
    for (i <- 0 until totalRound) {
      logger.info(s"Round: $i/$totalRound")
      val grads = downloadGrad(i)
      val nGrads = TreeUtils.expandGrads(grads, dataSet.length, nLabel)
      for (gID <- 0 until nLabel) {
        if (!labelEarlyStop(gID)) {
          val currTree = RegressionTree(dataSet, indices, nGrads(gID), i.toString)
          currTree.init()
          currTree.setLearningRate(learningRate).setMinChildSize(minChildSize)
          val continueBoosting = boostRound(i, currTree)
          if (!continueBoosting) labelEarlyStop(gID) = true
        }
      }
    }
  }
  def buildTree(tree: RegressionTree, continuous: Boolean): Unit = {

    while (tree.canGrow && tree.depth < maxDepth) {
      // Find best split in curr tree
      val bestLocalSplit = tree.findBestSplit()

      bestLocalSplit.setVersion(splitVersion)
      splitVersion += 1
      val bestSplit = getBestSplitFromServer(bestLocalSplit)
      if (bestSplit.gain != 0) {
        logger.debug(s"Global best split: $bestSplit")
      }

      val isLocalSplit = bestLocalSplit.getClientID == bestSplit.getClientID
      // If this split is in local dataset
      val updateCondition = if (continuous) {
        bestSplit.featureID == -1 || bestSplit.gain < 1e-6f
      } else bestSplit.featureID == -1
      if (updateCondition) {
        logger.debug(s"Set ${bestSplit.nodeID} as leaf")
        tree.setLeaf(tree.nodes(bestSplit.nodeID))
      } else {
        // update bestSplit from server to local tree
        tree.updateTree(bestSplit, isLocalSplit)
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
      gradData.putTensorMap("label", toFloatTensor(label))
    }
    // Upload
    if (flClient == null) {
      throw new IllegalArgumentException("FLClient not initialized.")
    }
    val response = flClient.fgbostStub.uploadLabel(gradData.build)
    Log4Error.invalidOperationError(response.getCode != 1, response.getResponse)
    logger.debug(response.getResponse)
  }

  /**
   * Download gradient from FLServer
   * @param treeID
   * @return
   */
  def downloadGrad(treeID: Int): Array[Array[Float]] = {
    // Note that g may be related to Y
    // H = 1 in regression
    val response = flClient.fgbostStub.downloadLabel("xgboost_grad", treeID)
    logger.debug("Downloaded grads from FLServer")
    val gradTable = response.getData
    val grad = getTensor("grad", gradTable).toArray
    val hess = getTensor("hess", gradTable).toArray
    curLoss = getTensor("loss", gradTable).value()
    Array(grad, hess)
  }

  def getBestSplitFromServer(split: Split): Split = {
    split.setClientID(flClient.getClientUUID)
    val response = flClient.fgbostStub.split(split.toDataSplit())
    if (response.getCode == 1) {
      logger.error(response.getResponse)
      throw new Exception("split failed, please check the log.")
    }
    else {
      val dataSplit = response.getSplit
      Split(
        dataSplit.getTreeID,
        dataSplit.getNodeID,
        dataSplit.getFeatureID,
        dataSplit.getSplitValue,
        dataSplit.getGain,
        dataSplit.getItemSetList,
        dataSplit.getVersion
      ).setClientID(dataSplit.getClientUid)
    }

  }

}

