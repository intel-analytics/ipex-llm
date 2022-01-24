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


import com.intel.analytics.bigdl.dllib.optim.{ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.base.StorageHolder
import com.intel.analytics.bigdl.ppml.common.{Aggregator, FLDataType, FLPhase}
import com.intel.analytics.bigdl.ppml.fgboost.common.{RMSEObjective, TreeObjective}
import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto._
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto._
import com.intel.analytics.bigdl.ppml.utils.ProtoUtils._
import org.apache.logging.log4j.LogManager

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

class FGBoostAggregator(validationMethods: Array[ValidationMethod[Float]] = null)
  extends Aggregator {

  val logger = LogManager.getLogger(this.getClass)

  var obj: TreeObjective = new RMSEObjective
  var nLabel = 1

  val serverTreeLeaves = new ArrayBuffer[Map[Int, Float]]()
  var validationSize = -1
  var basePrediction: Array[Float] = null
  val bestSplit = new java.util.HashMap[String, DataSplit]()
  var validationResult = Array[ValidationResult]()

  // wrapper methods to simplify data access
  def getLabelStorage() = aggregateTypeMap.get(FLPhase.LABEL).getTableStorage()
  def getSplitStorage() = aggregateTypeMap.get(FLPhase.SPLIT).getSplitStorage()
  def getTreeLeaveStorage() = aggregateTypeMap.get(FLPhase.TREE_LEAVES).getLeafStorage()
  def getEvalStorage() = aggregateTypeMap.get(FLPhase.EVAL).getBranchStorage()
  def getPredictStorage() = aggregateTypeMap.get(FLPhase.PREDICT).getBranchStorage()
  def getResultStorage() = aggregateTypeMap.get(FLPhase.RESULT).getTableStorage()

  override def initStorage(): Unit = {
    aggregateTypeMap.put(FLPhase.LABEL, new StorageHolder(FLDataType.TENSOR_MAP))
    aggregateTypeMap.put(FLPhase.RESULT, new StorageHolder(FLDataType.TENSOR_MAP))
    aggregateTypeMap.put(FLPhase.SPLIT, new StorageHolder(FLDataType.TREE_SPLIT))
    aggregateTypeMap.put(FLPhase.TREE_LEAVES, new StorageHolder(FLDataType.TREE_LEAVES))
    aggregateTypeMap.put(FLPhase.PREDICT, new StorageHolder(FLDataType.TREE_EVAL))
    aggregateTypeMap.put(FLPhase.EVAL, new StorageHolder(FLDataType.TREE_EVAL))
  }

  override def aggregate(flPhase: FLPhase): Unit = {
    // TODO aggregate split
    flPhase match {
      case FLPhase.LABEL => initGradient()
      case FLPhase.SPLIT => aggregateSplit()
      case FLPhase.TREE_LEAVES => aggregateTreeLeaves()
      case FLPhase.EVAL => aggEvaluate()
      case FLPhase.PREDICT => aggPredict()
      case _ => throw new NotImplementedError()
    }
  }

  def getBestSplit(treeID: String, nodeID: String): DataSplit = {
    synchronized(bestSplit) {
      val id = getTreeNodeId(treeID, nodeID)
      while ( {
        !bestSplit.containsKey(id)
      }) try bestSplit.wait()
      catch {
        case ie: InterruptedException =>
          throw new RuntimeException(ie)
      }
      return bestSplit.get(id)
    }
  }


  def setObj(objective: TreeObjective): Unit = {
    obj = objective
  }


  def setNumLabel(numLabel: Int): Unit = {
    nLabel = numLabel
  }


  def initGradient(): Unit = {
    val labelClientData = getLabelStorage().clientData
    val aggData = labelClientData.mapValues(_.getTensorsMap).values.flatMap(_.asScala)
      .map { data =>
        (data._1, data._2.getTensorList.asScala.toArray.map(_.toFloat))
      }.toMap
    val label = aggData("label")
    validationSize = label.length
    basePrediction = if (aggData.contains("predict")) {
      aggData("predict")
    } else {
      Array.fill[Float](validationSize)(0.5f)
    }
    val gradients = obj.getGradient(basePrediction, label)
    val grad = gradients(0)
    val hess = gradients(1)
    val loss = obj.getLoss(basePrediction, label)

    logger.info(s"====Loss ${loss} ====")
    logger.info("Label" + label.mkString("Array(", ", ", ")"))

    val metaData = MetaData.newBuilder()
      .setName("xgboost_grad")
      .setVersion(0)
      .build()
    val aggregatedModel = TensorMap.newBuilder()
      .setMetaData(metaData)
      .putTensors("grad", toFloatTensor(grad))
      .putTensors("hess", toFloatTensor(hess))
      .putTensors("predict", toFloatTensor(basePrediction))
      .putTensors("label", toFloatTensor(label))
      .build()
    // Update gradient
    getLabelStorage().clearClientAndUpdateServer(aggregatedModel)
    logger.debug("Init gradient completed.")
  }

  def setClientNum(clientNum: Int): this.type = {
    this.clientNum = clientNum
    this
  }

  def aggEvaluate(): Unit = {
    val aggPredict = aggregatePredict(FLPhase.EVAL)
    val newPredict = predictWithTree(aggPredict)
    val predictTensor = Tensor[Float](newPredict, Array(newPredict.size))
    val targetProto = getLabelStorage().serverData
    val targetTensor = getTensor("label", targetProto)
//    validationResult = validationMethods.map(vMethod => {
//      vMethod.apply(predictTensor, targetTensor)
//    })

    // Compute new residual
    updateGradient(newPredict)
  }

  def predictWithEncoding(encoding: Array[java.lang.Boolean], treeID: Int): Float = {
    val treeLeaves = serverTreeLeaves(treeID)
//    logger.debug("Predict with encoding" + encoding.mkString("Array(", ", ", ")"))
//    logger.debug("Tree map encoding" + treeLeaves.mkString("Array(", ", ", ")"))
    // go through the path by encoding to the leaf node
    var currIndex = 0
    while (!treeLeaves.contains(currIndex)) {
//      logger.debug("CurrIndex " + currIndex)
      if (currIndex > encoding.length - 1) {
        logger.error("Exception " + currIndex)
        logger.error("encoding " + encoding.mkString("Array(", ", ", ")"))
        logger.error("Leaves " + treeLeaves.toString)
      }
      if (encoding(currIndex).booleanValue()) {
        currIndex = currIndex * 2 + 1
      } else {
        currIndex = currIndex * 2 + 2
      }
    }
//    logger.debug("Reach leaf " + treeLeaves(currIndex))
    treeLeaves(currIndex)
  }


  def predictWithTree(aggPredict: Array[Array[(String, Array[java.lang.Boolean])]]): Array[Float] = {
    logger.info("Predict with new Tree")
    if (aggPredict.head.length == 1) {
      // Last tree
      aggPredict.map { predict =>
        predictWithEncoding(predict.last._2, serverTreeLeaves.length - 1)
      }
    } else {
      // n label tree or full tree
      aggPredict.flatMap { predict =>
        predict.map(x =>
          predictWithEncoding(x._2, x._1.toInt))
      }
    }
  }

  def aggregateTreeLeaves(): Unit = {
    logger.info(s"Add new Tree ${serverTreeLeaves.length}")
    val leafMap = getTreeLeaveStorage().clientData

    val treeIndexes = leafMap.values.head.getLeafIndexList.map(Integer2int).toArray
    val treeOutputs = leafMap.values.head.getLeafOutputList.map(Float2float).toArray
    val treeLeaves = treeIndexes.zip(treeOutputs).toMap
    // Add new tree leaves to server
    serverTreeLeaves += treeLeaves
    leafMap.clear()
  }

  def updateGradient(newPredict: Array[Float]): Unit = {
    // For XGBoost Regression with squared loss
    // g = y' - y, h = 1
    logger.info("Updating Gradient with new Predict")
    val tableStorage = getLabelStorage()
    val gradTable = tableStorage.serverData.getTensorsMap
    val predict = gradTable.get("predict").getTensorList.asScala.toArray.map(_.toFloat)
    val label = gradTable.get("label").getTensorList.asScala.toArray.map(_.toFloat)

    // Update Predict
    for (i <- predict.indices) {
      predict(i) = predict(i) + newPredict(i)
    }

    // Compute Gradients
    val gradients = obj.getGradient(predict, label)
    // Compute loss
    val loss = obj.getLoss(predict, label)

    logger.info(s"========Loss ${loss} =======")
    logger.debug("New Predict" + predict.mkString("Array(", ", ", ")"))
    logger.debug("New Grad" + gradients(0).mkString("Array(", ", ", ")"))

    val metaData = MetaData.newBuilder()
      .setName("xgboost_grad")
      .setVersion(tableStorage.version + 1)
      .build()
    val aggregatedModel = TensorMap.newBuilder()
      .setMetaData(metaData)
      .putTensors("grad", toFloatTensor(gradients(0)))
      .putTensors("predict", toFloatTensor(predict))
      .putTensors("hess", toFloatTensor(gradients(1)))
      .putTensors("label", toFloatTensor(label))
      .build()
    // Update gradient
    tableStorage.clearClientAndUpdateServer(aggregatedModel)
  }

  def getTreeNodeId(treeID: String, nodeID: String): String = treeID + "_" + nodeID
  def aggregateSplit(): Unit = {
    val splitMap = getSplitStorage().clientData
    var bestGain = Float.MinValue
    bestSplit.synchronized {
      splitMap.values.foreach { split =>
        if (split.getGain > bestGain) {
          val id = getTreeNodeId(split.getTreeID, split.getNodeID);
          bestSplit.put(id, split)
          bestGain = split.getGain
        }
      }
      splitMap.clear()
      bestSplit.notifyAll()
    }
  }

  def aggregatePredict(flPhase: FLPhase): Array[Array[(String, Array[java.lang.Boolean])]] = {
    // get proto and convert to scala object
    logger.info("Aggregate Predict")
    val boostEvalBranchMap = flPhase match {
      case FLPhase.EVAL => getEvalStorage().clientData
      case FLPhase.PREDICT => getPredictStorage().clientData
      case _ => throw new IllegalArgumentException()
    }
    val evalResults = boostEvalBranchMap.mapValues { list =>
      list.asScala.toArray.map { be =>
        be.getEvaluatesList.asScala.toArray.map { treePredict =>
          (treePredict.getTreeID, treePredict.getPredictsList.asScala.toArray)
        }
      }
    }
    val clientsIterator = boostEvalBranchMap.keys.toIterator
    val result = evalResults(clientsIterator.next())
    while (clientsIterator.hasNext) {
      result.zip(evalResults(clientsIterator.next())).foreach { be =>
        be._1.zip(be._2).foreach { case (clientPredict1, clientPredict2) =>
          require(clientPredict1._1 == clientPredict2._1, "Tree id miss match." +
            s" Got ${clientPredict1._1} ${clientPredict2._1}.")
          clientPredict1._2.indices.foreach { i =>
            clientPredict1._2(i) = clientPredict1._2(i) && clientPredict2._2(i)
          }
        }
      }
    }
    result
  }

  def aggPredict(): Unit = {
    val aggedPredict = aggregatePredict(FLPhase.PREDICT)
    val newPredict = aggedPredict.zip(basePrediction).map(p =>
      // Predict value of each boosting tree
      p._1.map(x => predictWithEncoding(x._2, x._1.toInt)).sum + p._2
    )
    val tableStorage = getResultStorage()
    val metaData = MetaData.newBuilder()
      .setName("predictResult")
      .setVersion(tableStorage.version)
      .build()
    val aggResult = TensorMap.newBuilder()
      .setMetaData(metaData)
      .putTensors("predictResult", toFloatTensor(newPredict)).build()
    tableStorage.clearClientAndUpdateServer(aggResult)
  }
}

